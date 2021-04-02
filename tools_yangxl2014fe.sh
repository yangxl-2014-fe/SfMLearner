#!/bin/bash

function header_error() {
  echo -e "\e[31m##################################################\e[39m"
  echo -e "\e[31m# $1\e[39m"
  echo -e "\e[31m##################################################\e[39m"
}

function header_warn() {
  echo -e "\e[33m##################################################\e[39m"
  echo -e "\e[33m# $1\e[39m"
  echo -e "\e[33m##################################################\e[39m"
}

function header_info() {
  echo -e "\e[32m##################################################\e[39m"
  echo -e "\e[32m# $1\e[39m"
  echo -e "\e[32m##################################################\e[39m"
}

function header_debug() {
  echo -e "\e[34m##################################################\e[39m"
  echo -e "\e[34m# $1\e[39m"
  echo -e "\e[34m##################################################\e[39m"
}

function text_error() {
  echo -e "\e[31m# $1\e[39m"
}

function text_warn() {
  echo -e "\e[33m# $1\e[39m"
}

function text_info() {
  echo -e "\e[32m# $1\e[39m"
}

function text_debug() {
  echo -e "\e[34m# $1\e[39m"
}
