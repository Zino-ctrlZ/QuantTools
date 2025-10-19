"""Helper decorators for Pydantic models."""

import sys
import traceback

def loud_post_init(fn):
    def wrap(self, ctx):
        try:
            return fn(self, ctx)
        except Exception as err:
            print(f"\n[model_post_init] {type(self).__name__} crashed:", file=sys.stderr)
            traceback.print_exception(type(err), err, err.__traceback__, file=sys.stderr)
            raise
    return wrap