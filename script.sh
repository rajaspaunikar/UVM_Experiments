echo "+uvm_ctrl" | sudo tee /sys/fs/cgroup/cgroup.subtree_control
sudo ./nuvmtop --no-tui --csv --outfile no_limit.csv --watch --poll-time 1000
echo "0 21474836480" | sudo tee /sys/fs/cgroup/default/uvm_ctrl.soft
echo "0 21474836480" | sudo tee /sys/fs/cgroup/default/uvm_ctrl.hard
sudo nuvmtop --no-tui --csv --outfile no_limit_one_process.csv --watch --poll-time 1000
