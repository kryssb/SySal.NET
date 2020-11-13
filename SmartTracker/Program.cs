using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Executables.SmartTracker
{
    class Program
    {
        static System.Text.RegularExpressions.Regex cls_rx = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*");

        static void Main(string[] args)
        {
            SySal.Processing.SmartTracking.SmartTracker Tk = new SySal.Processing.SmartTracking.SmartTracker();
            XmlSerializer xmls = new XmlSerializer(typeof(SySal.Processing.SmartTracking.Configuration));
            if (args.Length != 11)
            {
                Console.WriteLine("usage: SmartTracker <input cluster file> <XML configuration file> <pixel2micronx> <pixel2microny> <minx> <maxx> <miny> <maxy> <top|bottom> <output track file> <cycles>");
                Console.WriteLine("Cluster file syntax: LAYER Z X Y AREA");
                Console.WriteLine("N.B.: Z is expected to be the same for all clusters in a plane.");
                Console.WriteLine("Track file syntax: TRACK N AREASUM X Y Z AREA");
                Console.WriteLine("N.B.: for each track the first three fields are repeated at each grain.");
                Console.WriteLine("Typical configuration:");
                Console.WriteLine();
                xmls.Serialize(Console.Out, Tk.Config);
                return;
            }
#if DEBUG
            Console.WriteLine("You can attach a debugger now - RETURN to continue");
            Console.ReadLine();
#endif
            int cycles = Convert.ToInt32(args[10]);
            SySal.BasicTypes.Rectangle trackarea = new SySal.BasicTypes.Rectangle();
            trackarea.MinX = Convert.ToDouble(args[4]);
            trackarea.MaxX = Convert.ToDouble(args[5]);
            trackarea.MinY = Convert.ToDouble(args[6]);
            trackarea.MaxY = Convert.ToDouble(args[7]);
            SySal.BasicTypes.Vector2 pix2micron = new SySal.BasicTypes.Vector2();
            pix2micron.X = Convert.ToDouble(args[2]);
            pix2micron.Y = Convert.ToDouble(args[3]);
            bool istop;
            if (String.Compare(args[8], "top", true) == 0) istop = true;
            else if (String.Compare(args[8], "bottom", true) == 0) istop = false;
            else throw new Exception("Side must be top or bottom");
            Tk.Expose = true;
            System.IO.StreamReader r = new System.IO.StreamReader(args[0]);
            Tk.Config = (SySal.Processing.SmartTracking.Configuration)xmls.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(args[1])));
            Tk.TrackingArea = trackarea;
            Tk.Pixel2Micron = pix2micron;
            System.Collections.ArrayList planes = new System.Collections.ArrayList();
            System.Collections.ArrayList clusters = new System.Collections.ArrayList();
            string line;
            int linenum = -1;
            int i;
            int grs = 0;
            while ((line = r.ReadLine()) != null)
            {
                linenum++;
                System.Text.RegularExpressions.Match m = cls_rx.Match(line);
                if (m.Success == false || m.Length != line.Length)
                {
                    Console.WriteLine("Skipped line " + linenum + "\r\n\"" + line + "\"");
                    continue;
                }
                uint layer = Convert.ToUInt32(m.Groups[1].Value);
                SySal.Tracking.Grain2 g = new SySal.Tracking.Grain2();                
                g.Position.X = Convert.ToDouble(m.Groups[3].Value);
                g.Position.Y = Convert.ToDouble(m.Groups[4].Value);
                g.Area = Convert.ToUInt32(m.Groups[5].Value);
                if (layer >= planes.Count)
                {
                    for (i = planes.Count - 1; i < layer; i++)
                    {
                        planes.Add(new SySal.Tracking.GrainPlane());
                        clusters.Add(new System.Collections.ArrayList());
                    }
                }
                SySal.Tracking.GrainPlane gp = (SySal.Tracking.GrainPlane)planes[(int)layer];
                gp.Z = Convert.ToDouble(m.Groups[2].Value);
                ((System.Collections.ArrayList)clusters[(int)layer]).Add(g);
                grs++;
            }
            Console.WriteLine("Read " + grs + " cluster(s)");
            SySal.Tracking.GrainPlane[] tomography = (SySal.Tracking.GrainPlane[])planes.ToArray(typeof(SySal.Tracking.GrainPlane));
            for (i = 0; i < tomography.Length; i++)
                tomography[i].Grains = (SySal.Tracking.Grain2[])((System.Collections.ArrayList)clusters[i]).ToArray(typeof(SySal.Tracking.Grain2));
            clusters = null;
            planes = null;
            SySal.Tracking.Grain[][] tracks = null;
            Console.Write("Tracking...");
            if (cycles <= 1)
            {
                System.DateTime start = System.DateTime.Now;
                tracks = Tk.FindTracks(tomography, istop, 1000000, false, new SySal.BasicTypes.Vector2(), new SySal.BasicTypes.Vector2());
                Console.WriteLine("Completed in " + (System.DateTime.Now - start));
            }
            else
            {
                int c;
                System.DateTime start = System.DateTime.Now;
                for (c = 0; c <= cycles; c++)
                {
                    if (c == 1) start = System.DateTime.Now;
                    tracks = Tk.FindTracks(tomography, istop, 1000000, false, new SySal.BasicTypes.Vector2(), new SySal.BasicTypes.Vector2());
                }
                System.TimeSpan totaltime = System.DateTime.Now - start;
                Console.WriteLine("Completed in " + totaltime);
                Console.WriteLine("Cycles: " + cycles);
                Console.WriteLine("Average: " + System.TimeSpan.FromTicks(totaltime.Ticks / cycles) + "/cycle");
            }
            Console.WriteLine("Tracks found: " + tracks.Length);
            r.Close();
            System.IO.StreamWriter w = new System.IO.StreamWriter(args[9]);
            w.Write("TRACK N AREASUM X Y Z AREA");
            for (i = 0; i < tracks.Length; i++)
            {
                SySal.Tracking.Grain [] tkgrs = tracks[i];
                uint a = 0;
                foreach (SySal.Tracking.Grain ga in tkgrs) a += ga.Area;
                foreach (SySal.Tracking.Grain ga in tkgrs)
                {
                    w.WriteLine();
                    w.Write(i + "\t" + tkgrs.Length + "\t" + a + "\t" + ga.Position.X + "\t" + ga.Position.Y + "\t" + ga.Position.Z + "\t" + ga.Area);
                }
            }
            w.Flush();
            w.Close();
            System.Collections.ArrayList expinfo = Tk.ExposedInfo;
            foreach (SySal.BasicTypes.NamedParameter np in expinfo)
                Console.WriteLine(np.Name + " = " + np.Value);
        }
    }
}
