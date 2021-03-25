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

from abc import ABC, abstractmethod


class Field(ABC):

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def timestamp(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def chunksize(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def data(self):
        raise NotImplementedError()

    @abstractmethod
    def is_sorted(self):
        raise NotImplementedError()

    @abstractmethod
    def __bool__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def get_spans(self):
        raise NotImplementedError()
