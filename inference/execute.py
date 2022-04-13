import daemon

from InferenceDaemon import InferenceDaemon

with daemon.DaemonContext():
    InferenceDaemon().main()