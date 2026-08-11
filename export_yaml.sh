#!/usr/bin/env bash
# export_yaml.sh -- read a small YAML file and export its entries as shell variables.
#
#     source ./export_yaml.sh config_user.yaml
#
# MUST BE SOURCED, not executed: a child process cannot export into its parent. config.sh
# sources it as its last act, so a user's config_user.yaml overrides the defaults above it.
#
# THE SUBSET OF YAML UNDERSTOOD, which is all tools/config_wizard.py ever writes:
#
#     # comments, to end of line
#     section:                  # a bare key with no value: a heading, ignored
#       GPU: cuda               # KEY: value  -> export GPU=cuda
#       BATCH: 32
#       PRETRAIN_FLAGS: "--pretrain_steps 20000"     # quotes are stripped
#       SEED:                   # empty value: left alone, so the default stands
#
# Keys are shell variable names; indentation is decoration. No lists, no nesting beyond
# headings, no multi-line strings, no anchors. Anything else is reported and skipped rather
# than guessed at -- a configuration file that silently ignores half of what you wrote is
# worse than one that refuses it.
#
# PRECEDENCE. A variable that was already set IN THE ENVIRONMENT when config.sh started is
# never overwritten, so `GPU=cpu ./stage5_pretrain.sh` still wins over the YAML. config.sh
# records that set in ZETAGPT_ENV_SET before applying any default.
#
#     command-line flag  >  environment variable  >  config_user.yaml  >  config.sh

_yaml_file="${1:-}"
if [ -z "$_yaml_file" ] || [ ! -f "$_yaml_file" ]; then
  unset _yaml_file
  return 0 2>/dev/null || exit 0        # no file: the shipped defaults stand, silently
fi

_yaml_n=0
_yaml_bad=0
_yaml_lineno=0
while IFS= read -r _yaml_line || [ -n "$_yaml_line" ]; do
  _yaml_lineno=$((_yaml_lineno + 1))
  _yaml_line="${_yaml_line%%#*}"                                   # strip comments
  _yaml_line="${_yaml_line#"${_yaml_line%%[![:space:]]*}"}"        # trim leading space
  _yaml_line="${_yaml_line%"${_yaml_line##*[![:space:]]}"}"        # trim trailing space
  [ -z "$_yaml_line" ] && continue
  case "$_yaml_line" in
    *:*) ;;
    *)  echo "export_yaml: $_yaml_file:$_yaml_lineno: not 'KEY: value', skipped" >&2
        _yaml_bad=$((_yaml_bad + 1)); continue ;;
  esac
  _yaml_key="${_yaml_line%%:*}"
  _yaml_val="${_yaml_line#*:}"
  _yaml_val="${_yaml_val#"${_yaml_val%%[![:space:]]*}"}"           # trim leading space
  _yaml_val="${_yaml_val%"${_yaml_val##*[![:space:]]}"}"           # trim trailing space
  # a heading ("shared:") and an intentionally blank value both mean "nothing to set here"
  [ -z "$_yaml_val" ] && continue
  case "$_yaml_key" in
    [A-Za-z_]*) ;;
    *) echo "export_yaml: $_yaml_file:$_yaml_lineno: '$_yaml_key' is not a variable name, "\
"skipped" >&2; _yaml_bad=$((_yaml_bad + 1)); continue ;;
  esac
  case "$_yaml_val" in                                             # strip matching quotes
    \"*\") _yaml_val="${_yaml_val#\"}"; _yaml_val="${_yaml_val%\"}" ;;
    \'*\') _yaml_val="${_yaml_val#\'}"; _yaml_val="${_yaml_val%\'}" ;;
  esac
  # the environment set it before config.sh ran: leave it alone
  case " ${ZETAGPT_ENV_SET:-} " in
    *" $_yaml_key "*) continue ;;
  esac
  export "$_yaml_key=$_yaml_val"
  # remembered so config.sh's summary can say which knobs this file is responsible for
  ZETAGPT_YAML_SET="${ZETAGPT_YAML_SET:-} $_yaml_key"
  _yaml_n=$((_yaml_n + 1))
done < "$_yaml_file"

echo "config: $_yaml_n setting(s) from $_yaml_file" >&2
[ "$_yaml_bad" -gt 0 ] && echo "config: $_yaml_bad line(s) skipped in $_yaml_file" >&2
unset _yaml_file _yaml_line _yaml_key _yaml_val _yaml_n _yaml_bad _yaml_lineno
