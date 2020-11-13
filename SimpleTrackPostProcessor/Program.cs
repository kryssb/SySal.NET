using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Executables.SimpleTrackPostProcessor
{
    class Program
    {
        static System.Text.RegularExpressions.Regex tk_rx = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*");

        static void Main(string[] args)
        {
            SySal.Processing.SimpleTrackPostProcessing.SimpleTrackPostProcessor Pp = new SySal.Processing.SimpleTrackPostProcessing.SimpleTrackPostProcessor();
            XmlSerializer xmls = new XmlSerializer(typeof(SySal.Processing.SimpleTrackPostProcessing.Configuration));
            if (args.Length != 8)
            {
                Console.WriteLine("usage: SimpleTrackPostProcessor <input track grain file> <XML configuration file> <output track file> <zbase> <zext> <shrinkage> <correct (true | false)> <cycles>");
                Console.WriteLine("Track grain file syntax: TRACK N AREASUM X Y Z AREA");                
                Console.WriteLine("Track file syntax: TRACK N AREASUM X Y Z SX SY TOPZ BOTTOMZ SIGMA");
                Console.WriteLine("Typical configuration:");
                Console.WriteLine();
                xmls.Serialize(Console.Out, Pp.Config);
                return;
            }
#if DEBUG
            Console.WriteLine("You can attach a debugger now - RETURN to continue");
            Console.ReadLine();
#endif
            int cycles = Convert.ToInt32(args[7]);
            double zbase = Convert.ToDouble(args[3]);
            double zext = Convert.ToDouble(args[4]);
            double shrinkage = Convert.ToDouble(args[5]);
            bool correct = Convert.ToBoolean(args[6]);
            Pp.Expose = true;
            System.IO.StreamReader r = new System.IO.StreamReader(args[0]);
            Pp.Config = (SySal.Processing.SimpleTrackPostProcessing.Configuration)xmls.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(args[1])));
            System.Collections.ArrayList tracks = new System.Collections.ArrayList();
            string line;
            int linenum = -1;
            int i;
            int trackn;
            int lasttrackn = -1;
            bool first = true;
            int gcounter = 0;
            while ((line = r.ReadLine()) != null)
            {
                linenum++;
                System.Text.RegularExpressions.Match m = tk_rx.Match(line);
                if (m.Success == false || m.Length != line.Length)
                {
                    Console.WriteLine("Skipped line " + linenum + "\r\n\"" + line + "\"");
                    continue;
                }
                uint layer = Convert.ToUInt32(m.Groups[1].Value);
                SySal.Tracking.Grain g = new SySal.Tracking.Grain();             
                g.Position.X = Convert.ToDouble(m.Groups[4].Value);
                g.Position.Y = Convert.ToDouble(m.Groups[5].Value);
                g.Position.Z = Convert.ToDouble(m.Groups[6].Value);
                g.Area = Convert.ToUInt32(m.Groups[7].Value);
                trackn = Convert.ToInt32(m.Groups[1].Value);
                if (first || trackn != lasttrackn)
                {
                    tracks.Add(new SySal.Tracking.Grain[Convert.ToInt32(m.Groups[2].Value)]);
                    lasttrackn = trackn;
                    first = false;
                    gcounter = 0;
                }
                ((SySal.Tracking.Grain[])tracks[tracks.Count - 1])[gcounter++] = g;
            }
            Console.WriteLine("Read " + tracks.Count + " track(s)");
            SySal.Tracking.Grain[][] atracks = (SySal.Tracking.Grain[][])tracks.ToArray(typeof(SySal.Tracking.Grain[]));
            first = false;            
            for (i = 0; i < atracks.Length; i++)
            {
                SySal.Tracking.Grain[] tk = atracks[i];
                for (gcounter = 0; gcounter < tk.Length; gcounter++)
                    if (tk[gcounter] == null)
                    {
                        first = true;
                        Console.WriteLine("Missing grain #" + gcounter + " in track #" + i);
                    }
            }
            if (first)
            {
                Console.WriteLine("Wrong input, quitting.");
                return;
            }
            SySal.Tracking.MIPEmulsionTrack[] miptracks = null;
            Console.Write("Processing...");
            if (cycles <= 1)
            {
                System.DateTime start = System.DateTime.Now;
                miptracks = Pp.Process(atracks, zbase, zext, shrinkage, correct);
                Console.WriteLine("Completed in " + (System.DateTime.Now - start));
            }
            else
            {
                int c;
                System.DateTime start = System.DateTime.Now;
                for (c = 0; c <= cycles; c++)
                {
                    if (c == 1) start = System.DateTime.Now;
                    miptracks = Pp.Process(atracks, zbase, zext, shrinkage, correct);
                }
                System.TimeSpan totaltime = System.DateTime.Now - start;
                Console.WriteLine("Completed in " + totaltime);
                Console.WriteLine("Cycles: " + cycles);
                Console.WriteLine("Average: " + System.TimeSpan.FromTicks(totaltime.Ticks / cycles) + "/cycle");
            }
            Console.WriteLine("Tracks processed: " + miptracks.Length);
            r.Close();
            System.IO.StreamWriter w = new System.IO.StreamWriter(args[2]);
            w.WriteLine("TRACK N AREASUM IX IY IZ SX SY TZ BZ SIGMA");
            for (i = 0; i < miptracks.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo tk = miptracks[i].Info;
                w.WriteLine(i + " " + tk.Count + " " + tk.AreaSum + " " + tk.Intercept.X + " " + tk.Intercept.Y + " " + tk.Intercept.Z + " " + tk.Slope.X + " " + tk.Slope.Y + " " + tk.TopZ + " " + tk.BottomZ + " " + tk.Sigma);
            }
            w.Flush();
            w.Close();
            System.Collections.ArrayList expinfo = Pp.ExposedInfo;
            foreach (SySal.BasicTypes.NamedParameter np in expinfo)
                Console.WriteLine(np.Name + " = " + np.Value);
        }
    }
}
