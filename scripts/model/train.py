#%%

from analysis import make_label_table, VERSION, VERSION_TIMESTAMP

label_table = make_label_table(annotation_timestamp=VERSION_TIMESTAMP, root_version=VERSION)

#%%