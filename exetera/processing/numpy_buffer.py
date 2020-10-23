# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

class NumpyBuffer2:
    def __init__(self, dtype, block_pow=8):
        self.dtype_ = dtype
        self.list_ = list()

    def append(self, value):
        self.list_.append(value)

    def finalise(self):
        result = np.asarray(self.list_, dtype=self.dtype_)
        del self.list_
        return result


class NumpyBuffer:
    def __init__(self, dtype, block_pow=8):
        self.block_shift_ = block_pow
        self.block_size_ = 1 << block_pow
        self.block_mask_ = self.block_size_ - 1
        self.blocks_ = list()
        self.dtype_ = dtype
        self.current_block_ = None
        self.current_ = 0

    def append(self, value):
        if self.current_ == len(self.blocks_) * self.block_size_:
            self.blocks_.append(np.zeros(self.block_size_, dtype=self.dtype_))
            self.current_block_ = self.blocks_[-1]
        self.current_block_[self.current_ & self.block_mask_] = value
        self.current_ += 1


    def finalise(self):
        if self.current_ == 0:
            return None
        final = np.zeros(self.current_, dtype=self.dtype_)
        start = 0
        for b in range(len(self.blocks_) - 1):
            cur_block = self.blocks_[b]
            final[start:start + self.block_size_] = cur_block
            start += self.block_size_
            del cur_block
        current = self.current_ & self.block_mask_
        last_block = self.blocks_[-1]
        final[start:start + current] = last_block[0:current]
        del last_block

        self.blocks_ = list()
        self.current_ = 0

        return final


class ListBuffer:
    def __init__(self, block_pow=8):
        self.block_shift_ = block_pow
        self.block_size_ = 1 << block_pow
        self.block_mask_ = self.block_size_ - 1
        self.blocks_ = list()
        self.current_ = 0

    def append(self, value):
        if self.current_ == len(self.blocks_) * self.block_size_:
            self.blocks_.append([None] * self.block_size_)
        self.blocks_[self.current_ >> self.block_shift_][self.current_ & self.block_mask_] = value
        self.current_ += 1


    def finalise(self):
        if self.current_ == 0:
            return None
        final = [None] * self.current_
        start = 0
        for b in range(len(self.blocks_) - 1):
            cur_block = self.blocks_[b]
            final[start:start + self.block_size_] = cur_block
            start += self.block_size_
            del cur_block
        current = self.current_ & self.block_mask_
        last_block = self.blocks_[-1]
        final[start:start + current] = last_block[0:current]
        del last_block

        self.blocks_ = list()
        self.current_ = 0

        return final

# x = np.zeros(32)
# y = np.asarray([x for x in range(16)])
# start = 0
# inc = 16
# x[start:start+inc] = y
# print(y)