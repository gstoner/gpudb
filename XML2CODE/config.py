#!/usr/bin/python
"""
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

joinType = 0	# 0 for traditional hash join and 1 for invisible join
POS = 0			# 0 for MEM, 1 for PINNED and 2 for UVA
SOA = 0			# 0 for AOS and 1 for SOA
CODETYPE = 0 	# 0 for cuda, 1 for opencl
