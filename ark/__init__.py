import logging

# Library/application boundary: ark never configures logging for the
# process it's imported into (no basicConfig, no handlers beyond this
# no-op). A caller who wants to see ark's log output attaches their own
# handler to the "ark" logger (or a descendant of it).
logging.getLogger("ark").addHandler(logging.NullHandler())
