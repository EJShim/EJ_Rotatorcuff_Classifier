# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['D:\\projects\\EJ_ROTATORCUFF_CLASSIFIER'],
             binaries=[],
             datas=[('train_test_module/train_record_2block.npz', './train_test_module/'),
                    ('weights_build/2block_49', './weights_build/2block_49'),
                    ('icons','./icons'),
                    ('data/screen.png','./data')],
             hiddenimports=[],
             hookspath=['./'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='KRCTC',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon ='.\\icons\\kistlogo.ico' )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='KRCTC')
