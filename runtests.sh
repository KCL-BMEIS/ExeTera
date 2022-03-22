#! /bin/bash

do_coverage=false
do_report=false
do_html=false

# get the directory the script lives in and cd to it, might want this variable for later additions
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"

# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --coverage)
            do_coverage=true
        ;;
        --report)
            do_report=true
        ;;
        --html)
            do_html=true
        ;;
        --help)
            echo "runtests.sh [--coverage] [--report] [--html] [--help] FILES*"
            exit 0
        ;;
        *)
            # don't eat any file names passed on the command line
            break
        ;;
    esac
    shift
done

# run tests and other coverage operations if requested
if [ $do_coverage = true ]
then
    rm -f .coverage coverage.xml
    coverage run --omit=tests/*.py --branch -m unittest -c $*
    result=$?
    coverage xml
    
    if [ $do_report = true ]
    then
        coverage report
    fi

    if [ $do_html = true ]
    then
        rm -rf htmlcov
        coverage html
    fi
else
    python -m unittest -c $*
    result=$?
fi

exit $result
