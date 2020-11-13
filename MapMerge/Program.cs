using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.MapMerge
{
    class Program    
    {
        internal delegate double dFilterF(object o);

        internal class FilterF
        {
            public string Name;
            public dFilterF F;
            public string HelpText;
            public FilterF(string n, dFilterF f, string h) { Name = n; F = f; HelpText = h; }
        }

        internal static double fSegN(object o) { return (double)(short)((SySal.TotalScan.Segment)o).Info.Count; }
        internal static double fSegA(object o) { return (double)(int)((SySal.TotalScan.Segment)o).Info.AreaSum; }
        internal static double fSegS(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Sigma; }
        internal static double fSegSX(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Slope.X; }
        internal static double fSegSY(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Slope.Y; }
        internal static double fSegPX(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Intercept.X; }
        internal static double fSegPY(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Intercept.Y; }
        internal static double fSegPZ(object o) { return (double)((SySal.TotalScan.Segment)o).Info.Intercept.Z; }
        internal static double fSegLayer(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.Id; }
        internal static double fSegBrickId(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.BrickId; }
        internal static double fSegSheetId(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.SheetId; }
        internal static double fSegSide(object o) { return (double)((SySal.TotalScan.Segment)o).LayerOwner.Side; }
        internal static double fSegLayerPos(object o) { return (double)((SySal.TotalScan.Segment)o).PosInLayer; }
        internal static double fSegTrack(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null ? -1.0 : (double)s.TrackOwner.Id; }
        internal static double fSegTrackPos(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null ? -1.0 : (double)s.PosInTrack; }
        internal static double fSegTrackN(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null ? 0.0 : (double)s.TrackOwner.Length; }
        internal static double fSegUpVN(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Upstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Upstream_Vertex.Length; }
        internal static double fSegUpVID(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Upstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Upstream_Vertex.Id; }
        internal static double fSegUpVIP(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Upstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Upstream_Impact_Parameter; }
        internal static double fSegDownVN(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Downstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Downstream_Vertex.Length; }
        internal static double fSegDownVID(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Downstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Downstream_Vertex.Id; }
        internal static double fSegDownVIP(object o) { SySal.TotalScan.Segment s = (SySal.TotalScan.Segment)o; return s.TrackOwner == null || s.TrackOwner.Downstream_Vertex == null ? -1.0 : (double)s.TrackOwner.Downstream_Impact_Parameter; }

        static FilterF[] SegmentFilterFunctions = new FilterF[]
            { 
                new FilterF("N", fSegN, "Number of grains"),
                new FilterF("A", fSegA, "Total sum of the area in pixel"),
                new FilterF("S", fSegS, "Sigma"),
                new FilterF("SX", fSegSX, "X slope"),
                new FilterF("SY", fSegSY, "Y slope"),
                new FilterF("PX", fSegPX, "X position"),
                new FilterF("PY", fSegPY, "Y position"),
                new FilterF("PZ", fSegPZ, "Z position"),
                new FilterF("Brick", fSegBrickId, "Brick id"),
                new FilterF("Sheet", fSegSheetId, "Sheet id"),
                new FilterF("Side", fSegSide, "Side"),
                new FilterF("Layer", fSegLayer, "Layer id"),
                new FilterF("LPos", fSegLayerPos, "Position in layer"),
                new FilterF("Track", fSegTrack, "Track id"),
                new FilterF("TrackPos", fSegTrackPos, "Position in track"),
                new FilterF("NT", fSegTrackN, "Segments in the track"),
                new FilterF("UVID", fSegUpVID, "Id of the upstream vertex"),
                new FilterF("UVN", fSegUpVN, "Tracks at the upstream vertex"),
                new FilterF("UVIP", fSegUpVIP, "Upstream Impact Parameter of the owner track"),
                new FilterF("DVID", fSegDownVID, "Id of the downstream vertex"),
                new FilterF("DVN", fSegDownVN, "Tracks at the downstream vertex"),
                new FilterF("DVIP", fSegDownVIP, "Downstream Impact Parameter of the owner track")
            };

        internal class ObjFilter
        {
            NumericalTools.Function F;
            FilterF[] FMap;

            public ObjFilter(FilterF[] flist, string fstr)
            {
                int i;

                F = new NumericalTools.CStyleParsedFunction(fstr);
                FMap = new FilterF[F.ParameterList.Length];

                for (i = 0; i < FMap.Length; i++)
                {
                    string z = F.ParameterList[i];
                    foreach (FilterF ff1 in flist)
                        if (String.Compare(ff1.Name, z, true) == 0)
                        {
                            FMap[i] = ff1;
                            break;
                        }
                    if (FMap[i] == null) throw new Exception("Unknown parameter \"" + z + "\".");
                }

            }

            public bool Value(object o)
            {
                int p;
                for (p = 0; p < FMap.Length; p++)
                    F[p] = FMap[p].F(o);
                return F.Evaluate() != 0.0;
            }

        }


        const string swEditConfig = "/editconfigfile";
        const string swPosTol = "/postol";
        const string swSlopeTol = "/slopetol";
        const string swMaxOffset = "/maxoffset";
        const string swMinMatches = "/minmatches";
        const string swMapSize = "/mapsize";
        const string swQuick = "/quick";
        const string swRefInput = "/inr";
        const string swAddInput = "/ina";
        const string swOutput = "/out";
        const string swLog = "/log";
        const string swRefDataSet = "/rds";
        const string swAddDataSet = "/ads";
        const string swFilterDataSet = "/fds";
        const string swConfig = "/config";
        const string swFilter = "/filter";
        const string swRefBrick = "/rbk";
        const string swAddBrick = "/abk";

        static void Main(string[] args)
        {
            bool postolDefined = false;
            bool slopetolDefined = false;
            bool mapsizeDefined = false;
            bool maxposoffsetDefined = false;
            bool minmatchesDefined = false;
            bool quickDefined = false;            
            bool inputrefFileDefined = false;
            bool inputaddFileDefined = false;
            bool outputFileDefined = false;
            bool refdatasetDefined = false;
            bool adddatasetDefined = false;
            bool filterdatasetDefined = false;
            bool inputConfigDefined = false;
            bool logfileDefined = false;
            bool filterDefined = false;
            bool refbrickDefined = false;
            bool addbrickDefined = false;
            string FilterText = "";
            string RefFileIn = "";
            string AddFileIn = "";
            string OutFile = "";
            string RefDataSetName = "";
            string AddDataSetName = "";
            string FilterDataSetName = "";
            string ConfigFile = "";
            string LogFile = null;
            long RefBrick = 0;
            long AddBrick = 0;
            SySal.Processing.MapMerge.MapMerger MM = new SySal.Processing.MapMerge.MapMerger();
            SySal.Processing.MapMerge.Configuration C = (SySal.Processing.MapMerge.Configuration)MM.Config;
            SySal.Processing.MapMerge.MapManager.dMapFilter Filter = null;            
            try
            {
                int argnum;
                for (argnum = 0; argnum < args.Length; argnum++)
                {
                    switch (args[argnum].ToLower())
                    {
                        case swEditConfig:
                            {
                                System.Windows.Forms.Application.EnableVisualStyles();
                                System.Xml.Serialization.XmlSerializer xmls2 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MapMerge.Configuration));
                                try
                                {
                                    C = (SySal.Processing.MapMerge.Configuration)xmls2.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(args[argnum + 1])));
                                }
                                catch (Exception) { };
                                SySal.Processing.MapMerge.EditConfigForm ec = new SySal.Processing.MapMerge.EditConfigForm();
                                ec.C = C;
                                if (ec.ShowDialog() == DialogResult.OK)
                                {
                                    System.IO.StringWriter w = new System.IO.StringWriter();
                                    xmls2.Serialize(w, C);
                                    System.IO.File.WriteAllText(args[argnum + 1], w.ToString());                                 
                                }
                                return;
                            }
                        case swPosTol:
                            {
                                postolDefined = true;
                                C.PosTol = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case swSlopeTol:
                            {
                                slopetolDefined = true;
                                C.SlopeTol = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case swMaxOffset:
                            {
                                maxposoffsetDefined = true;
                                C.MaxPosOffset = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case swMapSize:
                            {
                                mapsizeDefined = true;
                                C.MapSize = System.Convert.ToDouble(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case swMinMatches:
                            {
                                minmatchesDefined = true;
                                C.MinMatches = System.Convert.ToInt32(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case swQuick:
                            {
                                quickDefined = true;
                                break;
                            }
                        case swFilter:
                            {
                                filterDefined = true;
                                FilterText = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swRefInput:
                            {
                                inputrefFileDefined = true;
                                RefFileIn = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swAddInput:
                            {
                                inputaddFileDefined = true;
                                AddFileIn = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swOutput:
                            {
                                outputFileDefined = true;
                                OutFile = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swConfig:
                            {
                                inputConfigDefined = true;
                                ConfigFile = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swRefDataSet:
                            {
                                refdatasetDefined = true;
                                RefDataSetName = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swAddDataSet:
                            {
                                adddatasetDefined = true;
                                AddDataSetName = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swFilterDataSet:
                            {
                                filterdatasetDefined = true;
                                FilterDataSetName = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swLog:
                            {
                                logfileDefined = true;
                                LogFile = args[argnum + 1];
                                argnum++;
                                break;
                            }
                        case swRefBrick:
                            {
                                refbrickDefined = true;
                                RefBrick = System.Convert.ToInt64(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        case swAddBrick:
                            {
                                addbrickDefined = true;
                                AddBrick = System.Convert.ToInt64(args[argnum + 1], System.Globalization.CultureInfo.InvariantCulture);
                                argnum++;
                                break;
                            }
                        default: throw new Exception("Unsupported switch: \"" + args[argnum] + "\".");
                    }
                }
                if (quickDefined) C.FavorSpeedOverAccuracy = true;
                if (filterDefined)
                    Filter = new SySal.Processing.MapMerge.MapManager.dMapFilter(new ObjFilter(SegmentFilterFunctions, FilterText).Value);
                if (inputConfigDefined)
                {
                    System.Xml.Serialization.XmlSerializer xmls1 = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MapMerge.Configuration));
                    System.IO.StreamReader r1 = new System.IO.StreamReader(ConfigFile);
                    C = (SySal.Processing.MapMerge.Configuration)xmls1.Deserialize(r1);
                    r1.Close();
                }
                if (inputrefFileDefined == false) throw new Exception("Reference volume must be defined.");
                if (refdatasetDefined == false) RefDataSetName = "TSR";
                if (inputaddFileDefined == false) throw new Exception("Please define the volume to be imported.");
                if (adddatasetDefined == false) throw new Exception("Please define the dataset name to assign to imported data.");
                if (outputFileDefined == false) throw new Exception("The output file must be defined.");            
            }
            catch (Exception x)
            {
                Console.WriteLine("Usage: MapMerge.exe {parameters}");
                Console.WriteLine("parameters");
                Console.WriteLine(swPosTol + " -> position tolerance (optional).");
                Console.WriteLine(swSlopeTol + " -> slope tolerance (optional).");
                Console.WriteLine(swMapSize + " -> map size (optional).");
                Console.WriteLine(swMaxOffset + " -> maximum position offset (optional).");
                Console.WriteLine(swMinMatches + " -> minimum number of matches per map (optional).");
                Console.WriteLine(swQuick + " -> favor speed over accuracy (optional).");
                Console.WriteLine(swRefInput + " -> reference volume.");
                Console.WriteLine(swRefDataSet + " -> dataset name to assign to reference data (default is \"TSR\").");
                Console.WriteLine(swRefBrick + " -> reset reference brick to specified brick number.");
                Console.WriteLine(swAddInput + " -> volume to import.");
                Console.WriteLine(swAddDataSet + " -> dataset name to assign to imported data.");
                Console.WriteLine(swFilterDataSet + " -> dataset name to filter imported data (leave blank for no filter).");
                Console.WriteLine(swAddBrick + " -> reset brick to assign to imported data.");
                Console.WriteLine(swOutput + " -> output file name.");
                Console.WriteLine(swLog + " -> log file (optional) - use \"con\" for console output.");
                Console.WriteLine(swConfig + " -> configuration file (optional).");
                Console.WriteLine(swFilter + " -> selection function for track matching (optional).");
                Console.WriteLine();
                Console.WriteLine("Variables that can be used for track filtering:");
                foreach (FilterF ff in SegmentFilterFunctions)
                    Console.WriteLine(ff.Name + " -> " + ff.HelpText);
                Console.WriteLine();
                Console.WriteLine(x.ToString());
                return;
            }
            System.IO.TextWriter logw = null;
            if (logfileDefined)
            {
                if (String.Compare(LogFile.Trim(), "con", true) == 0) logw = Console.Out;
                else logw = new System.IO.StreamWriter(LogFile);
            }            
            MM.Config = C;
            SySal.TotalScan.NamedAttributeIndex.RegisterFactory();
            SySal.TotalScan.BaseTrackIndex.RegisterFactory();
            SySal.TotalScan.MIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBNamedAttributeIndex.RegisterFactory();
            SySal.TotalScan.Flexi.Volume refv = new SySal.TotalScan.Flexi.Volume();
            SySal.TotalScan.Flexi.DataSet rds = new SySal.TotalScan.Flexi.DataSet();
            rds.DataId = RefBrick;
            rds.DataType = RefDataSetName;
            refv.ImportVolume(rds, (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(RefFileIn, typeof(SySal.TotalScan.Volume)));
            if (RefBrick > 0)
            {
                int n = refv.Layers.Length;
                int i;
                for (i = 0; i < n; i++)
                    if (refv.Layers[i].BrickId == 0)
                        ((SySal.TotalScan.Flexi.Layer)refv.Layers[i]).SetBrickId(RefBrick);
            }
            SySal.TotalScan.Flexi.DataSet ads = new SySal.TotalScan.Flexi.DataSet();
            ads.DataId = AddBrick;
            ads.DataType = AddDataSetName;
            SySal.TotalScan.Flexi.DataSet fds = new SySal.TotalScan.Flexi.DataSet();
            if (FilterDataSetName.Length > 0)
            {
                fds.DataId = AddBrick;
                fds.DataType = FilterDataSetName.Trim();
            }
            else fds = null;
            SySal.TotalScan.Flexi.Volume addv = new SySal.TotalScan.Flexi.Volume();
            addv.ImportVolume(ads, (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(AddFileIn, typeof(SySal.TotalScan.Volume)), fds);
            if (AddBrick > 0)
            {
                int n = addv.Layers.Length;
                int i;
                for (i = 0; i < n; i++)
                    if (addv.Layers[1].BrickId == 0)
                        ((SySal.TotalScan.Flexi.Layer)addv.Layers[i]).SetBrickId(AddBrick);
            }
            MM.AddToVolume(refv, addv, ads, null, Filter, logw);
            if (logw != null && logw != Console.Out)
            {
                logw.Flush();
                logw.Close();
                logw = null;
            }
            Console.WriteLine("Result written to " + SySal.OperaPersistence.Persist(OutFile, refv));            
        }
    }
}
