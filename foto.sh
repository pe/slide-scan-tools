#!/bin/bash
set -uo pipefail
set -x
# https://fotovideotec.de/dias_digitalisieren/
# https://www.so-fo.de/t22122f148-Sammlung-von-Kontaktbelegungen-bei-Diaprojektoren.html
#di4.sc4n?

# Zeug stoppen das Kamera blockiert
systemctl --user stop gvfs-daemon

# GPIO aktivieren
echo "24" > /sys/class/gpio/export
echo "25" > /sys/class/gpio/export
# als output
echo out >/sys/class/gpio/gpio24/direction
echo out >/sys/class/gpio/gpio25/direction

#Speichern auf SD-Karte
gphoto2 --set-config capturetarget=1
#Kommentar an
gphoto2 --set-config imagecommentenable=0

set -e

if [[ $# -lt 3 ]] ; then
    echo "$0 NUMBER_OF_DIAS_IN_MAGAZINE NUMER_OF_BLANKS_IN_MAGAZINE NAME_OF_MAGAZINE"
    exit 1
fi

readonly number="$1"
shift
readonly blanks="$1"
shift
#if [[ $number -gt 0 ]] ; then
#	echo -n "$*  " >> zeitpunkte.txt
#fi
gphoto2 --set-config imagecomment="$*"

for ((i=1;i<=number;i++)); do
    echo "1" > /sys/class/gpio/gpio24/value
    sleep 0.3
    echo "0" > /sys/class/gpio/gpio24/value
    sleep 1.5
    gphoto2 --trigger-capture
    sleep 0.5
done

for ((i=1;i<=blanks;i++)); do
    echo "1" > /sys/class/gpio/gpio24/value
    sleep 0.3
    echo "0" > /sys/class/gpio/gpio24/value
    sleep 1
done

echo "1" > /sys/class/gpio/gpio24/value
sleep 0.3
echo "0" > /sys/class/gpio/gpio24/value
