from trade.helpers.Logging import setup_logger
logger = setup_logger('trade.helpers.openbb_helper')

def load_openBB():
    import os
    from openbb import obb
    openbb_key = os.environ.get('OPENBB_KEY')
    if openbb_key is None:
        logger.critical("OPENBB_KEY environment variable not set. Some OpenBB Dependencies will not work.")
        return
    try:
        obb.account.login(pat=openbb_key, remember_me= True)
    except Exception as e:
        print("Error logging in to OpenBB:", e)
        pass
    obb.account.refresh()
    obb.account.save()