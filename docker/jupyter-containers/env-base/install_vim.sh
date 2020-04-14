DIST_NAME=`awk -F= '/^ID=/{print $2}' /etc/*-release | sed 's/\"//g'`
echo "OS Distro: $DIST_NAME"

case "$DIST_NAME" in
  "ubuntu") pkgmgr="apt-get" ;;
  "centos") pkgmgr="yum" ;;
esac

$pkgmgr install vim ctags -y


mkdir -p ~/.vim/colors
mkdir -p ~/.vim/bundle

wget https://raw.githubusercontent.com/pydemia/pydemia-theme/master/vim/.vim/colors/cobalt2.vim -O ~/.vim/colors/cobalt2.vim
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

wget https://raw.githubusercontent.com/pydemia/pydemia-theme/master/vim/.vimrc -O ~/.vimrc

#vim -u "~/.vimrc" +PlugInstall +qall > /dev/null
vim -c 'PluginInstall' -c 'qa!'

