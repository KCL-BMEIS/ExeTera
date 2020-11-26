
"""
Group / Field semantics
-----------------------

Location semantics
 * Fields can be created without a logical location. Such fields are written to a 'temp' location when required
 * Fields can be assigned a logical location or created with a logical location
 * Fields have a physical location at the point they are written to the dataset. Fields that are assigned to a logical
location are also guaranteed to be written to a physical location


"""


class Group:
    def __init__(self, parent):
        self.parent = parent

    def create_group(self, group_name):
        self.parent