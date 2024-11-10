
def load_openBB():
    import os
    from openbb import obb
    openbb_key = os.environ.get('OPENBB_KEY')
    obb.account.login(pat=openbb_key, remember_me= True)
    obb.account.refresh()
    obb.account.save()