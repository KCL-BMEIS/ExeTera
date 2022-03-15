#! /bin/bash

do_coverage=false
do_report=false

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
        --help)
            echo "runtests.sh [--coverage] [--report] [--help] FILES*"
            exit 0
        ;;
        *)
            # don't eat any file names passed on the command line
            break
        ;;
    esac
    shift
done

homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"

if [ $do_coverage = true ]
then
    rm -f .coverage
    coverage run -m unittest $*
    coverage xml
else
    python -m unittest $*
fi

if [ $do_report = true ]
then
    coverage report
fi
