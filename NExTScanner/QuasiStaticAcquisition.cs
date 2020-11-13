using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    public class QuasiStaticAcquisition
    {
        public const string SequenceString = "$SEQ$";
        public const string ImageString = "$IMAGE$";
        public const string TypeString = "$TYPE$";
        public const string SideString = "$SIDE$";
        public const string StripString = "$STRIP$";
        public const string TypeStringScan = "scan";

        public Sequence[] Sequences = new Sequence[0];
        public string FilePattern = "";

        public string ClusterMapFileName { get { return FilePattern.Replace(SequenceString, "").Replace(ImageString, "").Replace(TypeString, "cls"); } }

        public void WriteAllClusters()
        {
            System.IO.MemoryStream ms = new System.IO.MemoryStream();
            foreach (Sequence s in Sequences)
                s.WriteAllClusters(ms);
            /*
            System.IO.StreamWriter w = new System.IO.StreamWriter(ms);
            foreach (Sequence s in Sequences)
                s.WriteAllClusters(w);
            w.Flush();
            w.Close();
            */
            System.IO.File.WriteAllBytes(ClusterMapFileName, ms.ToArray());
        }

        [Serializable]
        public class PostProcessingInfo
        {
            public string Name;
            public string Settings;
        }

        [Serializable]
        public class Zone
        {
            public struct ProgressInfo
            {
                public uint BottomStripsReady;
                public uint TopStripsReady;
            };

            [NonSerialized]
            public ProgressInfo Progress;
            
            public uint Strips;
            public uint Views;
            public bool HasTop;
            public bool HasBottom;
            public SySal.BasicTypes.Rectangle ScanRectangle;
            public SySal.BasicTypes.Rectangle TransformedScanRectangle;
            public SySal.DAQSystem.Scanning.IntercalibrationInfo PlateMap;
            public SySal.BasicTypes.Vector ImageDelta;
            public SySal.BasicTypes.Vector2 ViewDelta;
            public SySal.BasicTypes.Vector2 StripDelta;
            public string ScanSettings;
            public string FileNameTemplate;
            public string OutputRWDFileName = "";

            public PostProcessingInfo[] PostProcessingSettings;

            static System.Xml.Serialization.XmlSerializer s_xmls = new System.Xml.Serialization.XmlSerializer(typeof(Zone));

            [NonSerialized]
            private string m_ZoneFileName;

            [NonSerialized]
            private string m_FileName;

            [NonSerialized]
            private string m_Dir;

            public Zone()
            {
                m_Dir = "";
                m_FileName = "";
            }

            public static Zone RecoverFromFile(string file)
            {
                Zone z = (Zone)s_xmls.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(file)));
                z.m_ZoneFileName = z.FileNameTemplate.Replace(QuasiStaticAcquisition.SequenceString, "").Replace(QuasiStaticAcquisition.SideString, "A").Replace(QuasiStaticAcquisition.StripString, "").Replace(QuasiStaticAcquisition.ImageString, "").Replace(QuasiStaticAcquisition.TypeString, QuasiStaticAcquisition.TypeStringScan);
                z.m_Dir = z.m_ZoneFileName.Substring(0, z.m_ZoneFileName.LastIndexOfAny(new char[] { '\\', '/' }));
                z.m_FileName = z.FileNameTemplate.Replace(QuasiStaticAcquisition.SequenceString, "").Replace(QuasiStaticAcquisition.SideString, "A").Replace(QuasiStaticAcquisition.StripString, "").Replace(QuasiStaticAcquisition.ImageString, "").Replace(QuasiStaticAcquisition.TypeString, "progress");
                return z;
            }

            public Zone(string filepattern)
            {
                FileNameTemplate = filepattern;
                m_ZoneFileName = filepattern.Replace(QuasiStaticAcquisition.SequenceString, "").Replace(QuasiStaticAcquisition.SideString, "A").Replace(QuasiStaticAcquisition.StripString, "").Replace(QuasiStaticAcquisition.ImageString, "").Replace(QuasiStaticAcquisition.TypeString, "scan");
                m_Dir = m_ZoneFileName.Substring(0, m_ZoneFileName.LastIndexOfAny(new char[] { '\\', '/' }));
                m_FileName = filepattern.Replace(QuasiStaticAcquisition.SequenceString, "").Replace(QuasiStaticAcquisition.SideString, "A").Replace(QuasiStaticAcquisition.StripString, "").Replace(QuasiStaticAcquisition.ImageString, "").Replace(QuasiStaticAcquisition.TypeString, "progress");
            }

            public void Update()
            {
                try
                {
                    if (m_ZoneFileName != null)
                    {
                        if (System.IO.Directory.Exists(m_Dir) == false)
                            System.IO.Directory.CreateDirectory(m_Dir);
                        System.IO.StringWriter swr = new System.IO.StringWriter();
                        s_xmls.Serialize(swr, this);
                        System.IO.File.WriteAllText(m_ZoneFileName, swr.ToString());
                        m_ZoneFileName = null;
                    }
                    string s = " " + this.Progress.TopStripsReady + " " + this.Progress.BottomStripsReady + " ";
                    System.IO.File.WriteAllText(m_FileName, s);                        
                }
                catch (Exception) { }
            }

            static System.Text.RegularExpressions.Regex rxProgress = new System.Text.RegularExpressions.Regex(@"\s+(\d+)\s+(\d+)\s+");

            public bool ReadProgress(ref uint topprogress, ref uint bottomprogress)
            {
                try
                {
                    var m = rxProgress.Match(System.IO.File.ReadAllText(m_FileName));
                    if (m.Success == false) return false;
                    topprogress = uint.Parse(m.Groups[1].Value);
                    bottomprogress = uint.Parse(m.Groups[2].Value);
                    return true;
                }
                catch (Exception) { return false; }
            }

            public void Abort()
            {
                try
                {
                    System.IO.File.Delete(m_FileName);
                }
                catch (Exception) { }
            }
        }

        public class Sequence
        {
            public uint Id;
            public uint Grains;
            public Layer[] Layers = new Layer[0];
            public QuasiStaticAcquisition Owner;

            public Sequence()
            {
                Id = 0;
            }

            static System.Text.RegularExpressions.Regex rx_Grains = new System.Text.RegularExpressions.Regex(@"\s*(?<SEQUENCE>\d+)\s+(?<FRAME0X>\S+)\s+(?<FRAME0Y>\S+)\s+(?<FRAME0Z>\S+)\s+(?<TOTAL>\d+)\s+(?<ID>\d+)\s+(?<VOLUME>\d+)\s+(?<FIRSTLAYER>\d+)\s+(?<LASTLAYER>\d+)\s+(?<X>\S+)\s+(?<Y>\S+)\s+(?<Z>\S+)\s+(?<TANX>\S+)\s+(?<TANY>\S+)\s+(?<TANZ>\S+)\s+(?<TX>\S+)\s+(?<TY>\S+)\s+(?<TZ>\S+)\s+(?<BX>\S+)\s+(?<BY>\S+)\s+(?<BZ>\S+)\s*");

            public string GrainFileName { get { return Owner.FilePattern.Replace(SequenceString, Id.ToString()).Replace(ImageString, "").Replace(TypeString, "grs"); } }

            public string ClusterMapFileName { get { return Owner.FilePattern.Replace(SequenceString, Id.ToString()).Replace(ImageString, "").Replace(TypeString, "cls"); } }

            public string ClusterMapFileNameWithInversion(bool invert, uint maxviews) 
            { 
                return Owner.FilePattern.Replace(SequenceString, (invert ? (maxviews - 1 - Id) : Id).ToString()).Replace(ImageString, "").Replace(TypeString, "cls"); 
            }

            public void WriteAllClusters(System.IO.StreamWriter w)
            {
                foreach (Layer lay in Layers)
                    lay.WriteClusters(lay.ClustersData, w);
            }

            public void WriteAllClusters(System.IO.MemoryStream ms)
            {
                foreach (Layer lay in Layers)
                    lay.WriteClusters(lay.ClustersData, ms);
            }

            public void WriteGrains(SySal.Imaging.Grain3D[] grs)
            {
                System.IO.StreamWriter w = null;
                try
                {
                    w = new System.IO.StreamWriter(GrainFileName);
                    w.WriteLine("SEQUENCE\tFRAME0X\tFRAME0Y\tFRAME0Z\tTOTAL\tID\tVOLUME\tFIRSTLAYER\tLASTLAYER\tX\tY\tZ\tTANX\tTANY\tTANZ\tTX\tTY\tTZ\tBX\tBY\tBZ");
                    int i;
                    for (i = 0; i < grs.Length; i++)
                    {
                        SySal.Imaging.Grain3D g = grs[i];
                        w.WriteLine(Id + "\t" +
                            Layers[0].Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            Layers[0].Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            Layers[0].Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            grs.Length + "\t" + i + "\t" +
                            g.Volume + "\t" + g.FirstLayer + "\t" + g.LastLayer + "\t" +
                            g.Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.Tangent.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.Tangent.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.Tangent.Z.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.TopExtent.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.TopExtent.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.TopExtent.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.BottomExtent.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.BottomExtent.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            g.BottomExtent.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    w.Flush();
                    w.Close();
                    this.Grains = (uint)grs.Length;
                }
                finally
                {
                    if (w != null) w.Close();
                }
            }

            public SySal.Imaging.Grain3D[] ReadGrains()
            {
                System.IO.StreamReader r = null;
                try
                {
                    r = new System.IO.StreamReader(GrainFileName);
                    string line;
                    System.Collections.ArrayList a = new System.Collections.ArrayList();
                    while ((line = r.ReadLine()) != null)
                    {
                        System.Text.RegularExpressions.Match m = rx_Grains.Match(line);
                        if (m.Success && m.Index == 0 && m.Length == line.Length)
                        {
                            SySal.Imaging.Grain3D g = new SySal.Imaging.Grain3D();
                            g.Clusters = new SySal.Imaging.Cluster3D[0];
                            g.Volume = uint.Parse(m.Groups["VOLUME"].Value);
                            g.FirstLayer = uint.Parse(m.Groups["FIRSTLAYER"].Value);
                            g.LastLayer = uint.Parse(m.Groups["LASTLAYER"].Value);
                            g.Position.X = double.Parse(m.Groups["X"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.Position.Y = double.Parse(m.Groups["Y"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.Position.Z = double.Parse(m.Groups["Z"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.Tangent.X = double.Parse(m.Groups["TANX"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.Tangent.Y = double.Parse(m.Groups["TANY"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.Tangent.Z = double.Parse(m.Groups["TANZ"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.TopExtent.X = double.Parse(m.Groups["TX"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.TopExtent.Y = double.Parse(m.Groups["TY"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.TopExtent.Z = double.Parse(m.Groups["TZ"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.BottomExtent.X = double.Parse(m.Groups["BX"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.BottomExtent.Y = double.Parse(m.Groups["BY"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            g.BottomExtent.Z = double.Parse(m.Groups["BZ"].Value, System.Globalization.CultureInfo.InvariantCulture);
                            a.Add(g);
                        }
                    }
                    Grains = (uint)a.Count;
                    return (SySal.Imaging.Grain3D[])a.ToArray(typeof(SySal.Imaging.Grain3D));
                }
                finally
                {
                    if (r != null) r.Close();
                }
            }

            public class Layer
            {
                public uint Id;
                public SySal.BasicTypes.Vector Position;
                public uint Clusters;
                public SySal.Imaging.Cluster[] ClustersData;
                public Sequence Owner;

                public Layer()
                {
                    Id = 0;
                    Clusters = 0;
                }

                public string ImageFileName { get { return Owner.Owner.FilePattern.Replace(SequenceString, Owner.Id.ToString()).Replace(ImageString, Id.ToString()).Replace(TypeString, "bmp"); } }
                public string ClusterMapFileName { get { return Owner.Owner.FilePattern.Replace(SequenceString, Owner.Id.ToString()).Replace(ImageString, Id.ToString()).Replace(TypeString, "cls"); } }

                static System.Text.RegularExpressions.Regex rx_Clusters = new System.Text.RegularExpressions.Regex(@"\s*(?<SEQUENCE>\d+)\s+(?<LAYER>\d+)\s+(?<TOTAL>\d+)\s+(?<STAGEX>\S+)\s+(?<STAGEY>\S+)\s+(?<STAGEZ>\S+)\s+(?<ID>\d+)\s+(?<CLSX>\S+)\s+(?<CLSY>\S+)\s+(?<AREA>\d+)\s+(?<IMXX>\S+)\s+(?<IMYY>\S+)\s+(?<IMXY>\S+)\s*");

                public SySal.Imaging.Cluster[] ReadClusters()
                {
                    System.IO.StreamReader r = null;
                    try
                    {
                        r = new System.IO.StreamReader(ClusterMapFileName);
                        string line;
                        System.Collections.ArrayList a = new System.Collections.ArrayList();
                        while ((line = r.ReadLine()) != null)
                        {
                            System.Text.RegularExpressions.Match m = rx_Clusters.Match(line);
                            if (m.Success && m.Index == 0 && m.Length == line.Length)
                            {
                                SySal.Imaging.Cluster c = new SySal.Imaging.Cluster();
                                c.Area = uint.Parse(m.Groups["AREA"].Value);
                                c.X = double.Parse(m.Groups["CLSX"].Value, System.Globalization.CultureInfo.InvariantCulture);
                                c.Y = double.Parse(m.Groups["CLSY"].Value, System.Globalization.CultureInfo.InvariantCulture);
                                c.Inertia.IXX = long.Parse(m.Groups["IMXX"].Value);
                                c.Inertia.IYY = long.Parse(m.Groups["IMYY"].Value);
                                c.Inertia.IXY = long.Parse(m.Groups["IMXY"].Value);
                                a.Add(c);
                            }                            
                        }
                        Clusters = (uint)a.Count;
                        return (SySal.Imaging.Cluster[])a.ToArray(typeof(SySal.Imaging.Cluster));
                    }
                    finally
                    {
                        if (r != null) r.Close();
                    }
                }

                public void WriteSummary()
                {
                    System.IO.StreamWriter w = new System.IO.StreamWriter(Owner.Owner.SummaryFileName, true);
                    if (w.BaseStream.Position == 0)                    
                        w.WriteLine("SEQ\tLAYER\tSTAGEX\tSTAGEY\tSTAGEZ\tCLUSTERS");
                    w.WriteLine(Owner.Id + "\t" + Id + "\t" +
                        Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                        Clusters.ToString());
                    w.Flush();
                    w.Close();
                }

                public void WriteClusters(SySal.Imaging.Cluster [] cls)
                {
                    System.IO.StreamWriter w = null;
                    try
                    {
                        System.IO.MemoryStream ms = new System.IO.MemoryStream();
                        w = new System.IO.StreamWriter(ms);
                        w.WriteLine("SEQUENCE\tLAYER\tTOTAL\tSTAGEX\tSTAGEY\tSTAGEZ\tID\tCLSX\tCLSY\tAREA\tIMXX\tIMYY\tIMXY");
                        int i;
                        for (i = 0; i < cls.Length; i++)
                        {
                            SySal.Imaging.Cluster c = cls[i];
                            w.WriteLine(Owner.Id + "\t" + Id + "\t" + cls.Length + "\t" +
                                Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                i + "\t" + 
                                c.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                c.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                c.Area + "\t" +
                                c.Inertia.IXX.ToString() + "\t" +
                                c.Inertia.IYY.ToString() + "\t" +
                                c.Inertia.IXY.ToString());
                        }
                        w.Flush();
                        w.Close();
                        System.IO.File.WriteAllBytes(ClusterMapFileName, ms.ToArray());
                        this.Clusters = (uint)cls.Length;
                    }
                    finally
                    {
                        if (w != null) w.Close();
                    }
                }

                public void WriteClusters(SySal.Imaging.Cluster[] cls, System.IO.StreamWriter w)
                {

                    if (w.BaseStream.Position == 0)
                        w.WriteLine("SEQUENCE\tLAYER\tTOTAL\tSTAGEX\tSTAGEY\tSTAGEZ\tID\tCLSX\tCLSY\tAREA\tIMXX\tIMYY\tIMXY");
                    int i;
                    for (i = 0; i < cls.Length; i++)
                    {
                        SySal.Imaging.Cluster c = cls[i];
                        w.WriteLine(Owner.Id + "\t" + Id + "\t" + cls.Length + "\t" +
                            Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            i + "\t" +
                            c.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            c.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            c.Area + "\t" +
                            c.Inertia.IXX.ToString() + "\t" +
                            c.Inertia.IYY.ToString() + "\t" +
                            c.Inertia.IXY.ToString());
                    }
                    this.Clusters = (uint)cls.Length;
                }

                public void WriteClusters(SySal.Imaging.Cluster[] cls, System.IO.MemoryStream ms)
                {
                    System.IO.BinaryWriter bw = new System.IO.BinaryWriter(ms);
                    bw.Write(Id);
                    bw.Write(cls.Length);
                    bw.Write(Position.X);
                    bw.Write(Position.Y);
                    bw.Write(Position.Z);
                    int i;
                    for (i = 0; i < cls.Length; i++)
                    {
                        SySal.Imaging.Cluster c = cls[i];
                        bw.Write(c.Area);
                        bw.Write(c.X);
                        bw.Write(c.Y);
                        bw.Write(c.Inertia.IXX);
                        bw.Write(c.Inertia.IXY);
                        bw.Write(c.Inertia.IYY);
                    }
                    this.Clusters = (uint)cls.Length;
                }
            }
        }

        static System.Text.RegularExpressions.Regex rx_filepattern = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*");

        public QuasiStaticAcquisition()
        {
            FilePattern = null;
        }

        public QuasiStaticAcquisition(string filepattern)
        {
            FilePattern = filepattern;
            string[] lines = System.IO.File.ReadAllLines(FilePattern.Replace(SequenceString, "").Replace(ImageString, "").Replace(TypeString, "sum"));
            System.Collections.Generic.Dictionary<uint, Sequence> sequences = new Dictionary<uint,Sequence>();
            foreach (string line in lines)
            {
                System.Text.RegularExpressions.Match m = rx_filepattern.Match(line);
                if (m.Success && m.Index == 0 && m.Length == line.Length)
                {
                    uint seqid = uint.Parse(m.Groups[1].Value);
                    Sequence seq = null;
                    if (sequences.ContainsKey(seqid)) seq = sequences[seqid];
                    else 
                    {
                        seq = new Sequence();
                        seq.Id = seqid;
                        seq.Owner = this;
                        sequences.Add(seqid, seq);
                    }
                    uint imgid = uint.Parse(m.Groups[2].Value);
                    Sequence.Layer layer = null;
                    foreach (Sequence.Layer ly in seq.Layers)
                        if (ly.Id == imgid)                        
                            throw new Exception("Duplicate layer found: sequence " + seqid + ", layer " + imgid);
                    Sequence.Layer [] nl = new Sequence.Layer[seq.Layers.Length + 1];
                    seq.Layers.CopyTo(nl, 0);
                    layer = new Sequence.Layer();
                    layer.Owner = seq;
                    layer.Id = imgid;
                    nl[seq.Layers.Length] = layer;
                    seq.Layers = nl;
                    layer.Position.X = double.Parse(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                    layer.Position.Y = double.Parse(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                    layer.Position.Z = double.Parse(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                    layer.Clusters = uint.Parse(m.Groups[6].Value);
                }
            }
            Sequences = new Sequence[sequences.Count];
            foreach (Sequence s in sequences.Values)
                Sequences[s.Id] = s;
            int i;
            for (i = 0; i < Sequences.Length; i++)
                if (Sequences[i] == null)
                    throw new Exception("Missing sequence entry " + i + " in QuasiStaticAcquisition.");
        }

        static public string PatternTemplate { get { return "acqset_" + SequenceString + "_" + ImageString + "." + TypeString; } }

        static public string IndexedPatternTemplate { get { return "acqset(" + SideString + "_" + StripString + ")_" + SequenceString + "_" + ImageString + "." + TypeString; } }

        public static string GetFilePattern(string indexedpattern, bool istop, uint strip)
        {
            return indexedpattern.Replace(SideString, istop ? "T" : "B").Replace(StripString, strip.ToString());
        }        

        public static string GetFocusDumpPattern(string indexedpattern)
        {
            return indexedpattern.Replace(SequenceString, "").Replace(ImageString, "").Replace(SideString, "").Replace(StripString, "").Replace(TypeString, "foc");
        }

        public string SummaryFileName { get { return FilePattern.Replace(SequenceString, "").Replace(ImageString, "").Replace(TypeString, "sum"); } }
    }
}
