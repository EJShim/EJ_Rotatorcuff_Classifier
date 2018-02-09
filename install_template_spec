# -*- mode: python -*-

block_cipher = None

a = Analysis(['main.py'],
             pathex=['D:\\projects\\EJ_ROTATORCUFF_CLASSIFIER'],
             binaries=[],
             datas=[                
                 ('icons', './icons'), 
                 ('weights_build', './weights_build'), 
                 ('data', './data'),                  
                 ('train_test_module/train_record_2block.npz', './train_test_module')
                ],
             hiddenimports=['scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='RCTclassifier',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main')
