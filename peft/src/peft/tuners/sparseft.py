# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class SparseFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        sparseft_type (`int`): Type of sparseft
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
    """
    sparseft_type: int = field(default=2, metadata={"help": "Type of SparseFT"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    
    def __post_init__(self):
        self.peft_type = PeftType.SPARSEFT


class SparseFTModel(torch.nn.Module):
    """
    Creates SparseFT model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the SparseFT model.

    Returns:
        `torch.nn.Module`: The SparseFT model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, SparseFTConfig >>> from peft import SparseFTModel, SparseFTConfig >>>
        config = SparseFTConfig(
            sparseft_type=2, target_modules=["q", "v"],
            budget=100000, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = SparseFTModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`SparseFTConfig`]): The configuration of the SparseFT model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_sparseft_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "sparseft_type": self.peft_config.sparseft_type,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)

                elif isinstance(target, torch.nn.Linear):
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_sparseft_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "delta_weight" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    else:
        raise NotImplementedError

class Linear(nn.Linear):
    # Simulates sparse finetuning of original model weights
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        sparseft_type : int = 0,
        fan_in_fan_out : bool = False, 
        **kwargs
    ):
        self.sparseft_type = sparseft_type
        self.fan_in_fan_out = fan_in_fan_out
        
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        # Actual trainable parameters
        if (sparseft_type == 0):
            self.mask_input = nn.Parameter(
                self.weight.new_zeros(in_features)
            )
            self.mask_output = nn.Parameter(
                self.weight.new_zeros(out_features)
            )
            self.delta_weight = nn.Parameter(
                self.weight.new_zeros((in_features, out_features))
            ) 
        elif (sparseft_type == 1):
            self.mask_weight = nn.Parameter(
                self.weight.new_zeros((in_features, out_features))
            )
            self.delta_weight = nn.Parameter(
                self.weight.new_zeros((in_features, out_features))
            )
        elif (sparseft_type == 2):
            # self.lora_A_weight = nn.Parameter(
            #     self.weight.new_zeros((10, in_features))
            # )
            # self.lora_A_bias = nn.Parameter(
            #     self.weight.new_zeros((4))
            # )
            # self.lora_B_weight = nn.Parameter(
            #     self.weight.new_zeros((out_features, 10))
            # )
            # self.lora_B_bias = nn.Parameter(
            #     self.weight.new_zeros((out_features))
            # )
            self.delta_weight = nn.Parameter(
                self.weight.new_zeros((out_features, in_features))
            )
        else:
            raise NotImplementedError
        
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if (self.sparseft_type == 0):
            if hasattr(self, 'mask_input'):
                nn.init.constant_(self.mask_input, 1.0)
                nn.init.constant_(self.mask_output, 1.0)
                nn.init.zeros_(self.delta_weight)
        elif (self.sparseft_type == 1):
            if hasattr(self, 'mask_weight'):
                nn.init.constant_(self.mask_weight, 1.0)
                nn.init.zeros_(self.delta_weight)
        elif (self.sparseft_type == 2):
            if hasattr(self, 'delta_weight'):
                # nn.init.normal_(self.lora_A_weight)
                # nn.init.normal_(self.lora_B_weight)
                nn.init.zeros_(self.delta_weight)
        else:
            raise NotImplementedError
        
        
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        # if (self.sparseft_type == 2):
        #     x = x*self.mult_vec+self.add_vec
        result = F.linear(x, T(self.weight), bias=self.bias)
        # if mask_input.requires grad has been set to False in run_glue, it mean we are running in discrete mode
        if (self.sparseft_type == 0):
            if (self.mask_input.requires_grad):
                inp = x*torch.sigmoid(self.mask_input)
                out = inp @ self.delta_weight
                out = out*torch.sigmoid(self.mask_output)
                result += out
            else:
                inp = x[..., self.mask_input]
                out = inp @ self.delta_weight
                result[..., self.mask_output] += out
        elif (self.sparseft_type == 1):
            delta_weight = torch.sigmoid(self.mask_weight)*self.delta_weight
            out = x @ delta_weight
            result += out
        elif (self.sparseft_type == 2):
            # res = F.linear(x, T(self.lora_A_weight))
            # res = F.linear(res, T(self.lora_B_weight))
            res = F.linear(x, T(self.delta_weight))
            result += res
        else:
            raise NotImplementedError
        return result

if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt):
        # Lora implemented in a dense layer
        def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            sparseft_type : int = 0,
            fan_in_fan_out : bool = False, 
            **kwargs
        ):
            self.sparseft_type = sparseft_type
            self.fan_in_fan_out = fan_in_fan_out
            
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
    
            # Actual trainable parameters
            if (sparseft_type == 0):
                self.mask_input = nn.Parameter(
                    self.weight.new_zeros(in_features)
                )
                self.mask_output = nn.Parameter(
                    self.weight.new_zeros(out_features)
                )
                self.delta_weight = nn.Parameter(
                    self.weight.new_zeros((in_features, out_features))
                ) 
            elif (sparseft_type == 1):
                self.mask_weight = nn.Parameter(
                    self.weight.new_zeros((in_features, out_features))
                )
                self.delta_weight = nn.Parameter(
                    self.weight.new_zeros((in_features, out_features))
                )
            elif (sparseft_type == 2):
                # self.lora_A_weight = nn.Parameter(
                #     self.weight.new_zeros((10, in_features))
                # )
                # self.lora_A_bias = nn.Parameter(
                #     self.weight.new_zeros((4))
                # )
                # self.lora_B_weight = nn.Parameter(
                #     self.weight.new_zeros((out_features, 10))
                # )
                # self.lora_B_bias = nn.Parameter(
                #     self.weight.new_zeros((out_features))
                # )
                self.delta_weight = nn.Parameter(
                    self.weight.new_zeros((out_features, in_features))
                )
            else:
                raise NotImplementedError

            self.reset_parameters()

        def reset_parameters(self):
            nn.Linear.reset_parameters(self)
            if (self.sparseft_type == 0):
                if hasattr(self, 'mask_input'):
                    nn.init.constant_(self.mask_input, 1.0)
                    nn.init.constant_(self.mask_output, 1.0)
                    nn.init.zeros_(self.delta_weight)
            elif (self.sparseft_type == 1):
                if hasattr(self, 'mask_weight'):
                    nn.init.constant_(self.mask_weight, 1.0)
                    nn.init.zeros_(self.delta_weight)
            elif (self.sparseft_type == 2):
                if hasattr(self, 'delta_weight'):
                    # nn.init.normal_(self.lora_A_weight)
                    # nn.init.normal_(self.lora_B_weight)
                    nn.init.zeros_(self.delta_weight)
            else:
                raise NotImplementedError

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                    result += output
                else:
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                    result += output
            return result

        def forward(self, x: torch.Tensor):
            def T(w):
                return w.T if self.fan_in_fan_out else w
            
            result = super().forward(x)

            # if mask_input.requires grad has been set to False in run_glue, it mean we are running in discrete mode
            if (self.sparseft_type == 0):
                if (self.mask_input.requires_grad):
                    inp = x*torch.sigmoid(self.mask_input)
                    out = inp @ self.delta_weight
                    out = out*torch.sigmoid(self.mask_output)
                    result += out
                else:
                    inp = x[..., self.mask_input]
                    out = inp @ self.delta_weight
                    result[..., self.mask_output] += out
            elif (self.sparseft_type == 1):
                delta_weight = torch.sigmoid(self.mask_weight)*self.delta_weight
                out = x @ delta_weight
                result += out
            elif (self.sparseft_type == 2):
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()

                    # res = F.linear(x, T(self.lora_A_weight))
                    # res = F.linear(res, T(self.lora_B_weight))
                    res = F.linear(x, T(self.delta_weight)).to(expected_dtype)
                    result += res
                else:
                    res = F.linear(x, T(self.delta_weight))
                    result += res
            else:
                raise NotImplementedError
            return result
