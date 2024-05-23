#!/bin/bash

YEAR=2024
VERSION=3450300

mkdir -p vendor/sqlite
curl -o sqlite-amalgamation.zip "https://www.sqlite.org/$YEAR/sqlite-amalgamation-$VERSION.zip"
unzip sqlite-amalgamation.zip
mv sqlite-amalgamation-$VERSION/* vendor/sqlite/
rmdir sqlite-amalgamation-$VERSION
rm sqlite-amalgamation.zip
