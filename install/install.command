dir=$(cd -- "$(dirname -- "$0")" && pwd)
rm ~/Library/Application\ Support/mari0
ln -s "$dir/mari0" ~/Library/Application\ Support
rm ~/Library/Application\ Support/mari0_libs
ln -s "$dir/mari0_libs" ~/Library/Application\ Support
