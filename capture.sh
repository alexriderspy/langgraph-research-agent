#!/bin/bash
name="$1"; shift
out="media/terminal/$name.log"
{ echo "sreemanti@MacBook-Pro langgraph-research-agent % $*"; "$@" 2>&1; } | tee "$out"
