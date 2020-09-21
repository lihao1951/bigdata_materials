# Linux设置Windows远程访问

1. Linux里运行`jupyter notebook --generate-config`
2. 在其中编辑`jupyter_notebook_config.py`
3. 修改各项属性值
   - c.NotebookApp.ip = '0.0.0.0'
   - c.NotebookApp.notebook_dir = '/home/**/jupyter/'
   - c.NotebookApp.open_browser = False
   - c.NotebookApp.password = ''
   - c.NotebookApp.port = 1538
4. 然后再输入`jupyter botebook password`，在命令行确认自己的密码
5. 在windows中打开` *.*.*.*:1538`,输入上一步确认的密码即可