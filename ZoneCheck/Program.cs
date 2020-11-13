using System;
using System.Collections.Generic;
using System.Text;
using ZoneStatus;

namespace ZoneCheck
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 5)
            {
                Console.WriteLine("Usage: ZoneCheck <Zone ID> <RWC input path> <TLG input path> <XML check output path> <XML monitoring output path>");
                return;
            }

            int series = Convert.ToInt32(args[0]);
            string rwcPath = args[1];
            string tlgPath = args[2];
            string qualityCheckFile = args[3];
            string monitoringFile = args[4];


            SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(tlgPath, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));

            StripLinkStatusInfo status = new StripLinkStatusInfo();
            StripMonitor stripMonitor = new StripMonitor();

            if (lz != null)
            {
                status.Check(rwcPath, lz, qualityCheckFile);
                stripMonitor.Fill(series, rwcPath, tlgPath, lz, monitoringFile);
            }

#if false
//            TODO: Comment from here to the end, just for debug
            if (StripLinkStatusInfo.IsScannedStripGood(qualityCheckFile, 0.0001))
                Console.WriteLine("OK");

            StripMonitorArrayClass StripMonitorArray = null;
            StripMonitorArray = new StripMonitorArrayClass(1);

            stripMonitor = new StripMonitor();
            stripMonitor.Read(monitoringFile);
            StripMonitorArray[0] = stripMonitor;
#endif
        }
    }
}
