for f in $(find . -name *.c -o -name *.h -o -name *.cpp -o -name *.hpp -o -name *.cc -o -name *.cu); do
    if [ -f $f ] ; then
        echo $f
        clang-format -i $f
        git add $f
    fi
done

