# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
# Derived from https://github.com/SalesforceAIResearch/CodeTree (Apache-2.0); modified for CodeHelm.
# Original License below.

# LICENSE HEADER MANAGED BY add-license-header
#
# /*
#  * Copyright (c) 2023, Salesforce, Inc.
#  * SPDX-License-Identifier: Apache-2
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */
#

import re
from typing import Optional


def parse_code_block(string: str, lang: str) -> Optional[str]:
    code = None
    # Allow optional whitespace/newline between the opening fence and the code,
    # and between the last line of code and the closing fence.
    code_pattern = fr"```{lang}\s*\n(.*?)\n?\s*```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        code = match.group(1)
    if code is None:
        generic_code_pattern = r"```\s*\n(.*?)\n?\s*```"
        match = re.search(generic_code_pattern, string, re.DOTALL)
        if match:
            code = match.group(1)
    if code is None:
        return parse_first_func(string, lang)
    # Remove unit tests written by itself
    return code


def parse_first_func(code: str, lang: str) -> Optional[str]:
    assert lang == "python", "Only python is supported for now. TODO: Rust"
    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    got_return = False
    for i, line in enumerate(code_lines):
        if line.startswith("def "):
            if def_i == -1:
                def_i = i
            else:
                break
        elif "return" in line and def_i != -1:
            got_return = True
        if line == "" and def_i != -1 and got_return:
            last_i = i
            break

    if last_i == 0:
        last_i = len(code_lines) - 1

    if def_i == -1:
        return ""

    return "\n".join(code_lines[def_i:last_i+1]).rstrip("[/PYTHON]")


def add_code_block(string: str, lang: str) -> str:
    return f"```{lang}\n{string}\n```"
