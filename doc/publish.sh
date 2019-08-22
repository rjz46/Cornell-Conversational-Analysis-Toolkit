#!/usr/bin/env bash
make html
rsync -r build/html/ $ZISSOU_USER@zissou.infosci.cornell.edu:/var/www/html/convokit/documentation