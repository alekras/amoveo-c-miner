cd /home/alexei/Erlang/Amoveo/amoveo-cuda-miner-AK-new
echo "`date`"
echo "from log:   device[0] = `grep 'Port\[0\].*found work' logs/miner.warning.log* | wc -l`"
echo "            device[1] = `grep 'Port\[1\].*found work' logs/miner.warning.log* | wc -l`"
echo "from debug: device[0] = `grep '!!!!!!!!!!' debug0.txt | wc -l`"
echo "            device[1] = `grep '!!!!!!!!!!' debug1.txt | wc -l`"
curl --stderr /dev/null -d '["account","BGv90RwK8L4OBSbl+6SUuyWSQVdkVDIOJY0i1wpWZINMTIBAM9/z3bOejY/LXm2AtA/Ibx4C7eeTJ+q0vhU9xlA="]' http://159.65.120.84:8085 | grep '"ok"' | awk -F, '{ print $4"]"$5 }' | awk -F] '{ print "veo="$1" shares="$2}'

