using System;

namespace SySal.Executables.ExtractLinkedIndex
{
    /// <summary>
    /// A command-line tool to produce lists of Ids of tracks to be ignored during alignment.
    /// </summary>
    /// <remarks>
    /// <para>ExtractLinkedIndex reads a TotalScan Reconstruction Volume and produces the list of tracks that are found to be linked on a 
    /// chosen sheet.</para>
    /// <para>The output list can be produced in two formats:
    /// <list type="bullet">
    /// <item><term>ASCII file (if the output path extension is not .TLG): in this case, the first line will contain the <c>Index</c> word, 
    /// and will be followed by lines each one containing one id.</term></item>
    /// <item><term>TLG file (requires a MultiSection TLG): in this case, the list of Ids is appended in a 
    /// <see cref="SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment"/> section.</term></item>. Notice that the TLG file must already exist.
    /// </list>
    /// </para>
    /// </remarks>
    public class Exe
    {
        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("Usage: ExtractLinkedIndex.exe <TSR OPERA persistence path> <sheet id> <output ASCII file>");
                Console.WriteLine("or: ExtractLinkedIndex.exe <TSR OPERA persistence path> <sheet id> <output TLG file>");
                return;
            }
            SySal.TotalScan.Volume v = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(args[0], typeof(SySal.TotalScan.Volume));
            int i, n;
            n = Convert.ToInt32(args[1]);            
            SySal.TotalScan.Layer layer = null;
            for (i = 0; i < v.Layers.Length; i++)
            {
                layer = v.Layers[i];
                if (layer.SheetId == n) break;
            }
            n = layer.Length;
            Console.WriteLine("Found layer " + layer.Id + " SheetId " + layer.SheetId);
            Console.WriteLine(n + " tracks found");
            int selected = 0;
            if (args[2].ToLower().EndsWith(".tlg"))
            {
                System.Collections.ArrayList ar = new System.Collections.ArrayList();
                for (i = 0; i < n; i++)
                {
                    SySal.TotalScan.Segment s = layer[i];
                    if (s.TrackOwner != null && s.TrackOwner.Length > 1)
                    {
                        ar.Add(i);
                        selected++;
                    }
                }
                SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment ai = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment();
                ai.Ids = (int[])ar.ToArray(typeof(int));
                SySal.OperaPersistence.Persist(args[2], ai);
            }
            else
            {
                System.IO.StreamWriter w = new System.IO.StreamWriter(args[2], false);
                w.WriteLine("Index");                
                for (i = 0; i < n; i++)
                {
                    SySal.TotalScan.Segment s = layer[i];
                    if (s.TrackOwner != null && s.TrackOwner.Length > 1)
                    {
                        w.WriteLine(i);
                        selected++;
                    }
                }
                w.Flush();
                w.Close();
            }
            Console.WriteLine(selected + " tracks selected");
        }
    }
}
