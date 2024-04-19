if [ "$#" -ne 2 ]
then
  echo "Usage: source run_in_tmux_window.sh <window_name> <command>"
  return 0
fi

# echo $1
# echo $2

# create window if necessary
if ! tmux select-window -t $1 &> /dev/null; then
  tmux new-window -n $1
fi

# switch back to original window
tmux last-window

# run command
tmux send -t "$1" "$2" ENTER
