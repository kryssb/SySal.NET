using System;
using System.Collections.Generic;
using System.Text;

namespace SySal
{
    public class OpEmuRecHelper
    {
        public class OpEmuRecDS : SySal.TotalScan.Flexi.DataSet
        {
            public OpEmuRecDS()
            {
                this.DataId = 0;
                this.DataType = "OpEmuRec"; 
            }

            public static OpEmuRecDS TheOpEmuRecDS = new OpEmuRecDS();
        }

        public class OpEmuRecVolume : SySal.TotalScan.Flexi.Volume
        {            
            public void AttachSegmentsToLayers()
            {
                int[] laycounts = new int[Layers.Length];
                int i, j;
                for (i = 0; i < Tracks.Length; i++)
                {
                    SySal.TotalScan.Flexi.Track tk = (SySal.TotalScan.Flexi.Track)Tracks[i];
                    for (j = 0; j < tk.Length; j++)
                        laycounts[tk[j].LayerOwner.Id]++;
                }
                for (i = 0; i < laycounts.Length; i++)
                {
                    ((OpEmuRecLayer)Layers[i]).CreateSegments(laycounts[i]);
                    laycounts[i] = 0;
                }
                for (i = 0; i < Tracks.Length; i++)
                {
                    SySal.TotalScan.Flexi.Track tk = (SySal.TotalScan.Flexi.Track)Tracks[i];
                    for (j = 0; j < tk.Length; j++)
                        ((OpEmuRecLayer)Layers[tk[j].LayerOwner.Id]).SetSegment(laycounts[tk[j].LayerOwner.Id]++, (OpEmuRecSegment)tk[j]);
                }
            }

            public void LinkTrackToVertex(int tkid, int vtxid, bool tkisupstream)
            {
                if (tkisupstream) m_Tracks[tkid].SetDownstreamVertex(m_Vertices[vtxid]);
                else m_Tracks[tkid].SetUpstreamVertex(m_Vertices[vtxid]);
                m_Vertices[vtxid].AddTrack(m_Tracks[tkid], tkisupstream);
            }

            public class OpEmuRecSegment : SySal.TotalScan.Flexi.Segment, SySal.Executables.EasyReconstruct.IExtendedSegmentInformation
            {
                public static OpEmuRecSegment s_LastSegment = null;

                public override object Clone()
                {
                    return new OpEmuRecSegment(this);
                }

                protected OpEmuRecSegment(OpEmuRecSegment s) : base(s, s.DataSet) 
                {
                    m_ExtendedFields = s.m_ExtendedFields;
                    m_ExtendedValues = s.m_ExtendedValues;
                }

                public override string ToString()
                {
                    string a = base.ToString() + "\r\nExtended Fields";
                    int i;
                    for (i = 0; i < m_ExtendedFields.Length; i++)                    
                        a += "\r\n" + m_ExtendedFields[i] + " = " + m_ExtendedValues[i].ToString();
                    return a;
                }

                public OpEmuRecSegment(SySal.TotalScan.Layer layer, SySal.Tracking.MIPEmulsionTrackInfo info, SySal.TotalScan.Index ix, SySal.TotalScan.Flexi.DataSet ds)
                    : base(new SySal.TotalScan.Segment(info, ix), ds)
                {
/*
                    this.m_Info = info;
                    this.m_Index = ix;
 */
                    this.m_LayerOwner = layer;
                    this.m_PosInLayer = -1;

                    s_LastSegment = this;
                }

                public OpEmuRecSegment(SySal.TotalScan.Layer layer, SySal.Tracking.MIPEmulsionTrackInfo info, SySal.TotalScan.Flexi.DataSet ds) : base(new SySal.TotalScan.Segment(info, new SySal.TotalScan.NullIndex()), ds)
                {
/*
                    this.m_Info = info;
                    this.m_Index = new SySal.TotalScan.NullIndex();
 */ 
                    this.m_LayerOwner = layer;
                    this.m_PosInLayer = -1;

                    s_LastSegment = this;
                }

                public void SetLayer(OpEmuRecLayer lay, int posinlayer)
                {
                    this.m_LayerOwner = lay;
                    this.m_PosInLayer = posinlayer;
                }

                #region IExtendedSegmentInformation Members

                string[] m_ExtendedFields = new string[0];
                object[] m_ExtendedValues = new object[0];

                public string[] ExtendedFields
                {
                    get { return m_ExtendedFields; }
                }

                public object ExtendedField(string name)
                {
                    int i;
                    for (i = 0; i < m_ExtendedFields.Length && String.Compare(m_ExtendedFields[i], name, true) != 0; i++) ;
                    if (i == m_ExtendedFields.Length) throw new Exception("Unknown field name \"" + name + "\".");
                    return m_ExtendedValues[i];
                }

                public Type ExtendedFieldType(string name)
                {
                    return ExtendedField(name).GetType();
                }

                #endregion

                public void SetExtendedField(string name, object value)
                {                    
                    int i;
                    for (i = 0; i < m_ExtendedFields.Length && String.Compare(m_ExtendedFields[i], name, true) != 0; i++) ;
                    if (i == m_ExtendedFields.Length)
                    {
                        string[] newn = new string[m_ExtendedFields.Length + 1];
                        m_ExtendedFields.CopyTo(newn, 0);
                        newn[i] = name;
                        m_ExtendedFields = newn;
                        object[] newo = new object[m_ExtendedValues.Length + 1];
                        m_ExtendedValues.CopyTo(newo, 0);
                        newo[i] = value;
                        m_ExtendedValues = newo;
                    }                    
                }
            }

            public class OpEmuRecLayer : SySal.TotalScan.Flexi.Layer
            {
                public OpEmuRecLayer(int id, SySal.BasicTypes.Vector refc, double minz, double maxz) : base(id, 0, 0, 0)
                {                    
                    m_DownstreamZ = maxz;
                    m_DownstreamZ_Updated = true;
                    m_Id = id;
                    m_RefCenter = refc;
                    m_SheetId = 0;
                    m_Side = 0;
                    m_UpstreamZ = minz;
                    m_UpstreamZ_Updated = true;
                    SySal.TotalScan.AlignmentData a = new SySal.TotalScan.AlignmentData();
                    a.AffineMatrixXX = a.AffineMatrixYY = 1.0;
                    a.AffineMatrixXY = a.AffineMatrixYX = 0.0;
                    a.DShrinkX = a.DShrinkY = 1.0;
                    a.SAlignDSlopeX = a.SAlignDSlopeY = 0.0;
                    a.TranslationX = a.TranslationY = a.TranslationZ = 0.0;
                    this.SetAlignmentData(a);
                }

                public void CreateSegments(int size)
                {
                    if (this.Segments == null || this.Segments.Length == 0)
                    {
                        this.Segments = new SySal.TotalScan.Segment[size];
                    }
                    else
                    {
                        SySal.TotalScan.Segment[] segs = new SySal.TotalScan.Segment[this.Segments.Length + size];
                        this.Segments.CopyTo(segs, 0);
                        this.Segments = segs;
                    }
                }

                public void SetSegment(int id, OpEmuRecSegment seg)
                {
                    this.Segments[id] = seg;
                    seg.SetLayer(this, id);
                }

                public void SetZs(double z, double downz, double upz, long brick, int plate, int side)
                {
                    m_RefCenter.Z = z;
                    m_DownstreamZ = downz;
                    m_DownstreamZ_Updated = true;
                    m_UpstreamZ = upz;
                    m_UpstreamZ_Updated = true;
                    m_SheetId = plate;
                    m_Side = (short)side;
                    m_BrickId = brick;
                }                
            }

            private class OpEmuRecLayerList : SySal.TotalScan.Flexi.Volume.LayerList
            {
                public OpEmuRecLayerList(SySal.BasicTypes.Vector refc, double minz, double maxz, int layers)
                {
                    this.Items = new SySal.TotalScan.Layer[layers];
                    int i;
                    for (i = 0; i < layers; i++)
                        this.Items[i] = new OpEmuRecLayer(i, refc, minz, maxz);
                }
            }

            private class OpEmuRecVertexList : SySal.TotalScan.Flexi.Volume.VertexList
            {
                public OpEmuRecVertexList()
                {
                    this.Items = new SySal.TotalScan.Vertex[0];
                }

                public OpEmuRecVertexList(int vtxcount)
                {
                    this.Items = new SySal.TotalScan.Vertex[vtxcount];
                    int i;
                    for (i = 0; i < vtxcount; i++)
                        this.Items[i] = new SySal.TotalScan.Flexi.Vertex(OpEmuRecDS.TheOpEmuRecDS, i);
                }
            }

            private class OpEmuRecTrackList : SySal.TotalScan.Flexi.Volume.TrackList
            {
                public OpEmuRecTrackList(int el)
                {
                    this.Items = new SySal.TotalScan.Track[el];
                    while (--el >= 0)
                    {
                        this.Items[el] = new SySal.TotalScan.Flexi.Track(OpEmuRecDS.TheOpEmuRecDS, el);
                    }
                }                
            }

            public OpEmuRecVolume(SySal.BasicTypes.Vector refc, double minz, double maxz, int layers, int tracks)
            {
                this.m_Extents.MaxX = this.m_Extents.MinX = refc.X;
                this.m_Extents.MaxY = this.m_Extents.MinY = refc.Y;
                this.m_Extents.MinZ = minz;
                this.m_Extents.MaxZ = maxz;
                this.m_RefCenter = refc;
                this.m_Layers = new OpEmuRecLayerList(refc, minz, maxz, layers);
                this.m_Tracks = new OpEmuRecTrackList(tracks);
                this.m_Vertices = new OpEmuRecVertexList();
            }
        }

        public static void CreateOpEmuRecVolume(double x, double y, double z, double minz, double maxz, int layers, int tracks)
        {
            SySal.BasicTypes.Vector r = new SySal.BasicTypes.Vector();
            r.X = x;
            r.Y = y;
            r.Z = z;
            Vol = new OpEmuRecVolume(r, minz, maxz, layers, tracks);
        }

        public static void SetSegmentDS(string type, long id)
        {
            SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
            ds.DataType = type;
            ds.DataId = id;
            SegDS = ds;
        }

        public static int InsertLayer(double z, double downz, double upz, int plate, int side)
        {            
            int i;
            for (i = 0; i < Vol.Layers.Length && (plate != Vol.Layers[i].SheetId || side != Vol.Layers[i].Side); i++);
            if (i < Vol.Layers.Length) return i;            
            SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
            v.X = Vol.RefCenter.X;
            v.Y = Vol.RefCenter.Y;
            v.Z = z;
            OpEmuRecVolume.OpEmuRecLayer ly = new OpEmuRecVolume.OpEmuRecLayer(0, v, upz, downz);
            ly.SetZs(z, downz, upz, 0, plate, side);
            ((SySal.TotalScan.Flexi.Volume.LayerList)Vol.Layers).Insert(ly);            
            for (i = 0; i < Vol.Layers.Length && (plate != Vol.Layers[i].SheetId || side != Vol.Layers[i].Side); i++) ;            
            return i;
        }

        public static int GetNumberOfSegmentsInLayer(int layer)
        {
            return Vol.Layers[layer].Length;
        }        

        public static void SetLayerZ(int id, double z, double downz, double upz, int plate, int side)
        {
            ((OpEmuRecVolume.OpEmuRecLayer)(Vol.Layers[id])).SetZs(z, downz, upz, BrickId, plate, side);
        }

        public static void SetupMomentumGeometry(double meas_err, double lead_radlen, double emul_radlen, double plastic_radlen, double csbox_radlen, double interbrick_radlen, int lastcs)
        {
            int i;
            for (i = 0; i < Vol.Layers.Length; i++)
                ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[i]).SetRadiationLength(emul_radlen);
            ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[0]).SetDownstreamRadiationLength(interbrick_radlen);
            ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[Vol.Layers.Length - 1]).SetUpstreamRadiationLength(interbrick_radlen);
            for (i = 1; i < Vol.Layers.Length; i++)
                if (Vol.Layers[i - 1].Side == 1)
                {
                    ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[i - 1]).SetDownstreamRadiationLength(lead_radlen);
                    ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[i - 1]).SetUpstreamRadiationLength(plastic_radlen);
                    ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[i]).SetDownstreamRadiationLength(plastic_radlen);
                    ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[i]).SetUpstreamRadiationLength(lead_radlen);
                }
            if (lastcs >= 0)
            {
                ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[lastcs]).SetUpstreamRadiationLength(csbox_radlen);
                if (lastcs < Vol.Layers.Length - 1) ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[lastcs + 1]).SetDownstreamRadiationLength(csbox_radlen);
            }
            SySal.Executables.EasyReconstruct.MomentumFitForm.MCSLikelihood = new SySal.Processing.MCSLikelihood.MomentumEstimator();
            SySal.Processing.MCSLikelihood.Configuration cfg = (SySal.Processing.MCSLikelihood.Configuration)SySal.Executables.EasyReconstruct.MomentumFitForm.MCSLikelihood.Config;
            cfg.ConfidenceLevel = 0.90;
            cfg.MinimumMomentum = 0.05;
            cfg.MaximumMomentum = 100.0;
            cfg.MinimumRadiationLengths = 0;
            cfg.MomentumStep = 0.05;
            cfg.Name = "Interactive";
            cfg.SlopeError = meas_err;
            cfg.Geometry = new SySal.TotalScan.Geometry(Vol.Layers);
            SySal.Executables.EasyReconstruct.MomentumFitForm.MCSLikelihood.Config = cfg;
        }

        public static void AddSegment(int track, short grains, int areasum, double px, double py, double sx, double sy, double sigma, int layerid, long zoneid, int mutkid)
        {
            SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
            SySal.TotalScan.Layer layer = Vol.Layers[layerid];
            info.Count = (ushort)grains;
            info.AreaSum = (uint)areasum;
            info.Intercept.X = px;
            info.Intercept.Y = py;
            info.Intercept.Z = layer.RefCenter.Z;
            info.Slope.X = sx;
            info.Slope.Y = sy;
            info.Slope.Z = 1.0;
            info.Sigma = sigma;
            info.TopZ = layer.DownstreamZ;
            info.BottomZ = layer.UpstreamZ;
            Vol.Tracks[track].AddSegment(new OpEmuRecVolume.OpEmuRecSegment(layer, info, new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(zoneid, layer.Side, mutkid), SegDS));
        }

        static void SetExtendedIntInLastSegment(string name, int value)
        {
            OpEmuRecVolume.OpEmuRecSegment.s_LastSegment.SetExtendedField(name, value);
        }

        static void SetExtendedDoubleInLastSegment(string name, double value)
        {
            OpEmuRecVolume.OpEmuRecSegment.s_LastSegment.SetExtendedField(name, value);
        }

        static void SetExtendedStringInLastSegment(string name, string value)
        {
            OpEmuRecVolume.OpEmuRecSegment.s_LastSegment.SetExtendedField(name, value);
        }

        static int callsAddSegmentToLayer = 0;

        static SySal.TotalScan.Flexi.DataSet SegDS = OpEmuRecDS.TheOpEmuRecDS;

        public static void AddSegmentToLayer(int posinlayer, short grains, int areasum, double px, double py, double sx, double sy, double sigma, int layerid, long zoneid, int mutkid)
        {
            callsAddSegmentToLayer++;
            SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
            SySal.TotalScan.Layer layer = Vol.Layers[layerid];
            info.Count = (ushort)grains;
            info.AreaSum = (uint)areasum;
            info.Intercept.X = px;
            info.Intercept.Y = py;
            info.Intercept.Z = layer.RefCenter.Z;
            info.Slope.X = sx;
            info.Slope.Y = sy;
            info.Slope.Z = 1.0;
            info.Sigma = sigma;
            info.TopZ = layer.DownstreamZ;
            info.BottomZ = layer.UpstreamZ;            
            ((OpEmuRecVolume.OpEmuRecLayer)layer).SetSegment(posinlayer, new OpEmuRecVolume.OpEmuRecSegment(layer, info, new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(zoneid, layer.Side, mutkid), SegDS));
        }

        static int callsAddTaggedSegmentToLayer = 0;

        public static void AddTaggedSegmentToLayer(int posinlayer, short grains, int areasum, double px, double py, double sx, double sy, double sigma, int layerid, long zoneid, int mutkid)
        {
            callsAddTaggedSegmentToLayer++;
            SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
            SySal.TotalScan.Layer layer = Vol.Layers[layerid];
            info.Count = (ushort)grains;
            info.AreaSum = (uint)areasum;
            info.Intercept.X = px;
            info.Intercept.Y = py;
            info.Intercept.Z = layer.RefCenter.Z;
            info.Slope.X = sx;
            info.Slope.Y = sy;
            info.Slope.Z = 1.0;
            info.Sigma = sigma;
            info.TopZ = layer.DownstreamZ;
            info.BottomZ = layer.UpstreamZ;
            ((OpEmuRecVolume.OpEmuRecLayer)layer).SetSegment(posinlayer, new OpEmuRecVolume.OpEmuRecSegment(layer, info, new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(zoneid, layer.Side, mutkid), SegDS));
        }

        public static void CreateSegments(int layer, int count)
        {            
            ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[layer]).CreateSegments(count);
        }

        public static int CreateTracks(int count)
        {
            int ret = Vol.Tracks.Length;
            SySal.TotalScan.Flexi.Track[] tk = new SySal.TotalScan.Flexi.Track[count];
            int i;
            for (i = 0; i < tk.Length; i++)
                tk[i] = new SySal.TotalScan.Flexi.Track(SegDS, ret + i);
            ((SySal.TotalScan.Flexi.Volume.TrackList)Vol.Tracks).Insert(tk);
            return ret;
        }

        public static int CreateVertices(int count)
        {
            int ret = Vol.Vertices.Length;
            SySal.TotalScan.Flexi.Vertex [] v = new SySal.TotalScan.Flexi.Vertex[count];
            int i;
            for (i = 0; i < v.Length; i++)
                v[i] = new SySal.TotalScan.Flexi.Vertex(SegDS, ret + i);
            ((SySal.TotalScan.Flexi.Volume.VertexList)Vol.Vertices).Insert(v);
            return ret;
        }

        public static void FindVertices(string vconfig)
        {
            AORec.Clear();
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.AlphaOmegaReconstruction.Configuration));
            SySal.Processing.AlphaOmegaReconstruction.Configuration aorecconfig = (SySal.Processing.AlphaOmegaReconstruction.Configuration)xmls.Deserialize(new System.IO.StringReader(vconfig));            
            aorecconfig.VtxAlgorithm = SySal.Processing.AlphaOmegaReconstruction.VertexAlgorithm.Global;
            aorecconfig.VtxFitWeightEnable = false;
            AORec.Config = aorecconfig;
            Vol = AORec.RecomputeVertices(Vol);
        }

        public static void LinkTrackToVertex(int tkid, int vtxid, bool trackisupstream)
        {
            ((OpEmuRecVolume)Vol).LinkTrackToVertex(tkid, vtxid, trackisupstream);
        }

        public static void SetAlignment(int layerid, double mxx, double mxy, double myx, double myy, double tx, double ty, double msx, double msy, double dsx, double dsy)
        {
            SySal.TotalScan.AlignmentData al = new SySal.TotalScan.AlignmentData();
            al.AffineMatrixXX = mxx;
            al.AffineMatrixXY = mxy;
            al.AffineMatrixYX = myx;
            al.AffineMatrixYY = myy;
            al.TranslationX = tx;
            al.TranslationY = ty;
            al.TranslationZ = 0.0;
            al.SAlignDSlopeX = dsx;
            al.SAlignDSlopeY = dsy;
            al.DShrinkX = msx;
            al.DShrinkY = msy;
            ((OpEmuRecVolume.OpEmuRecLayer)Vol.Layers[layerid]).SetAlignment(al);
        }

        internal static SySal.Processing.MCSAnnecy.MomentumEstimator MCSAnnecy = new SySal.Processing.MCSAnnecy.MomentumEstimator();

        public static string MomentumDumpFile = "";

        public static void SetMomentumDebugInfo(int tkid, double MC_momentum, int MC_pdg)
        {
            SySal.TotalScan.Track tk = Vol.Tracks[tkid];
            tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("_MC_momentum_"), MC_momentum);
            tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("_MC_pdg_"), (double)MC_pdg);
        }

        public static void ComputeMomentumAnnecy(int tkid)
        {
            new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.MCSAnnecy.Configuration)).Serialize(Console.Out, MCSAnnecy.Config);
            MCSAnnecy.DiffLog = Console.Out;
            MCSAnnecy.FitLog = Console.Out;
            int i;
            SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
            ds.DataId = 0;
            ds.DataType = "$TEMP$";
            for (i = ((tkid < 0) ? 0 : tkid); i < ((tkid < 0) ? Vol.Tracks.Length : (tkid + 1)); i++)
            {
                SySal.TotalScan.Track tk = Vol.Tracks[i];
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("P"), -99.0);
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMIN"), -99.0);
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMAX"), -99.0);
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PCL"), 0.9);
                tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("P", 0), -99.0);
                tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PMIN", 0), -99.0);
                tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PMAX", 0), -99.0);
                tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PCL", 0), 0.9);
                SySal.TotalScan.Flexi.Track ftk = new SySal.TotalScan.Flexi.Track(ds, tk.Id);
                int j;
                for (j = 0; j < tk.Length; j++)
                {
                    SySal.TotalScan.Flexi.Segment seg = new SySal.TotalScan.Flexi.Segment(tk[j], ds);
                    seg.SetLayer(tk[j].LayerOwner, tk[j].PosInLayer);
                    ftk.AddSegments(new SySal.TotalScan.Flexi.Segment[1] { seg });
                }
                try
                {
                    SySal.TotalScan.MomentumResult mr = MCSAnnecy.ProcessData(ftk.BaseTracks);
                    tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("P"), mr.Value);
                    tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMIN"), mr.LowerBound);
                    tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PMAX"), mr.UpperBound);
                    tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PCL"), mr.ConfidenceLevel);
                    tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("P", 0), mr.Value);
                    tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PMIN", 0), mr.LowerBound);
                    tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PMAX", 0), mr.UpperBound);
                    tk.SetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PCL", 0), mr.ConfidenceLevel);
                    if (MomentumDumpFile != null && MomentumDumpFile.Length > 0)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo[] basetracks = ftk.BaseTracks;
                        double mc_momentum = -1.0;
                        int mc_pdg = 0;
                        try
                        {
                            mc_momentum = tk.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("_MC_momentum_"));
                            mc_pdg = (int)tk.GetAttribute(new SySal.TotalScan.NamedAttributeIndex("_MC_pdg_"));
                        }
                        catch (Exception) { }
                        System.IO.File.AppendAllText(MomentumDumpFile, "\r\nTrack " + i + " " + basetracks.Length + " " + tk.Length + " " + mc_momentum + " " + mr.Value + " " + mc_pdg + " " + tk.ListAttributes().Length);
                        if (basetracks != null)
                        {
                            System.IO.File.AppendAllText(MomentumDumpFile, "\r\nBasetracks");
                            int ibi;
                            for (ibi = 0; ibi < basetracks.Length; ibi++)
                                System.IO.File.AppendAllText(MomentumDumpFile, "\r\n" + basetracks[ibi].Field + " " + basetracks[ibi].Slope.X + " " + basetracks[ibi].Slope.Y + " " + basetracks[ibi].Intercept.X + " " + basetracks[ibi].Intercept.Y + " " + basetracks[ibi].Intercept.Z);                            
                        }
                        else System.IO.File.AppendAllText(MomentumDumpFile, "\r\nNo basetracks.");
                        System.IO.File.AppendAllText(MomentumDumpFile, "\r\nMicrotracks");
                        {
                            int imi;
                            for (imi = 0; imi < tk.Length; imi++)
                                System.IO.File.AppendAllText(MomentumDumpFile, "\r\n" + tk[imi].LayerOwner.SheetId + " " + tk[imi].LayerOwner.Side + " " + tk[imi].Info.Slope.X + " " + tk[imi].Info.Slope.Y + " " + tk[imi].Info.Intercept.X + " " + tk[imi].Info.Intercept.Y + " " + tk[imi].Info.Intercept.Z);
                        }
                    }
                }
                catch (Exception)
                {
                    /*
                    try { tk.RemoveAttribute(new SySal.TotalScan.NamedAttributeIndex("P")); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.TotalScan.NamedAttributeIndex("PMIN")); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.TotalScan.NamedAttributeIndex("PMAX")); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.TotalScan.NamedAttributeIndex("PCL")); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("P", 0)); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PMIN", 0)); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PMAX", 0)); }
                    catch (Exception) { };
                    try { tk.RemoveAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex("PCL", 0)); }
                    catch (Exception) { };
                     */
                }
            }
        }

        public static void SetSegmentExtendedFields(string extendedfieldspec)
        {
            SySal.Executables.EasyReconstruct.MainForm.SetSegmentExtendedFields(extendedfieldspec);
        }

        public static bool RunDisplay(string selection)
        {
            /*
            SySal.TotalScan.Flexi.Volume myv = new SySal.TotalScan.Flexi.Volume();            
            myv.ImportVolume(OpEmuRecDS.TheOpEmuRecDS, Vol);
            Vol = myv;            
             */
            SySal.Executables.EasyReconstruct.MainForm.RunDisplay(Vol as SySal.TotalScan.Flexi.Volume/*myv*/, selection);
            Vol = SySal.Executables.EasyReconstruct.MainForm.EditedVolume;
            if (Vol != null && ViewMaps == null)
            {
                int i, layerid;
                int totalviews = 0;
                ViewMaps = new ViewMap[Vol.Layers.Length][];                
                for (layerid = 0; layerid < Vol.Layers.Length; layerid++)
                {                    
                    SySal.TotalScan.Layer lay = Vol.Layers[layerid];
                    System.Collections.ArrayList vm = new System.Collections.ArrayList();                    
                    for (i = 0; i < lay.Length; i++)
                    {
                        SySal.TotalScan.Segment seg = lay[i];
                        ViewMap nv = new ViewMap(seg);
                        int ifs = vm.BinarySearch(nv, nv);
                        if (ifs < 0) vm.Insert(~ifs, nv);                        
                        else ((ViewMap)vm[ifs]).Segments.Add(seg);
                    }
                    for (i = 0; i < vm.Count; i++)
                        ((ViewMap)vm[i]).Segments.TrimToSize();
                    ViewMaps[layerid] = (ViewMap[])vm.ToArray(typeof(ViewMap));                    
                    totalviews += ViewMaps[layerid].Length;
                }                
            }            
            return Vol != null;
        }

        internal static string Uid;

        internal static long BrickId;

        internal static string ScratchPath;

        internal static SySal.Executables.BatchLink.Exe LinkerExe;

        internal static SySal.Processing.AlphaOmegaReconstruction.AlphaOmegaReconstructor AORec;

        private static NumericalTools.CStyleParsedFunction m_AlignIgnoreFilter;

        private delegate double dFVar(SySal.Scanning.MIPBaseTrack mb);

        private static double fN(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.Count; }
        private static double fA(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.AreaSum; }
        private static double fId(SySal.Scanning.MIPBaseTrack mb) { return mb.Id; }
        private static double fPX(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.Intercept.X; }
        private static double fPY(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.Intercept.Y; }
        private static double fSX(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.Slope.X; }
        private static double fSY(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.Slope.Y; }
        private static double fS(SySal.Scanning.MIPBaseTrack mb) { return mb.Info.Sigma; }        

        private static dFVar [] AIFMap = new dFVar[0];

        internal static bool IgnoreQ(SySal.Scanning.MIPBaseTrack mb)
        {
            if (m_AlignIgnoreFilter == null) return false;
            int i;
            for (i = 0; i < AIFMap.Length; i++)

                m_AlignIgnoreFilter[i] = AIFMap[i](mb);
            return m_AlignIgnoreFilter.Evaluate() != 0.0;
        }

        public static string AlignIgnoreFilter
        {
            set
            {
                if (value == null)
                {
                    m_AlignIgnoreFilter = null;
                    AIFMap = new dFVar[0];
                }
                else
                {
                    m_AlignIgnoreFilter = new NumericalTools.CStyleParsedFunction(value);
                    AIFMap = new dFVar[m_AlignIgnoreFilter.ParameterList.Length];
                    int i;
                    for (i = 0; i < AIFMap.Length; i++)
                        switch (m_AlignIgnoreFilter.ParameterList[i].ToUpper())
                        {
                            case "N": AIFMap[i] = new dFVar(fN); break;
                            case "A": AIFMap[i] = new dFVar(fA); break;
                            case "ID": AIFMap[i] = new dFVar(fId); break;
                            case "PX": AIFMap[i] = new dFVar(fPX); break;
                            case "PY": AIFMap[i] = new dFVar(fPY); break;
                            case "SX": AIFMap[i] = new dFVar(fSX); break;
                            case "SY": AIFMap[i] = new dFVar(fSY); break;
                            case "S": AIFMap[i] = new dFVar(fS); break;
                            default: throw new Exception("Unsupported variable: " + m_AlignIgnoreFilter.ParameterList[i]);
                        }
                }
            }
        }

        class PlateInfo : IDisposable
        {
            public double Z;
            public bool IsUpdated = false;
            public string ReaderFile;
            public string TLGFile;
            public int PlateId;
            public System.IO.FileStream ReaderStream;
            public System.IO.BinaryWriter bW;
            int Total = 0;
            internal int Side1Total = 0, Side2Total = 0;
            SySal.BasicTypes.Rectangle Rect;
            public SySal.Scanning.PostProcessing.SlopeCorrections SlopeCorr;

            ~PlateInfo()
            {
                Free();
            }
            public PlateInfo(int plateid, double z)
            {
                Z = z;
                PlateId = plateid;
                ReaderFile = OpEmuRecHelper.ScratchPath + "_" + PlateId + ".reader";
                TLGFile = OpEmuRecHelper.ScratchPath + "_" + PlateId + ".tlg";                
                ReaderStream = new System.IO.FileStream(ReaderFile, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.Read);                
                bW = new System.IO.BinaryWriter(ReaderStream);
                UpdateReader();
            }

            void Free()
            {
                if (ReaderStream != null)
                {
                    ReaderStream.Close();
                    ReaderStream = null;
                }
                if (ReaderFile != null)
                    try
                    {
                        System.IO.File.Delete(ReaderFile);
                    }
                    catch (Exception) { }
                if (TLGFile != null)
                    try
                    {
                        System.IO.File.Delete(TLGFile);
                    }
                    catch (Exception) { }
            }

            #region IDisposable Members

            public void Dispose()
            {
                Free();
                GC.SuppressFinalize(this);
            }

            #endregion

            internal void AddMicrotrack(long zone, short side, int id, int id_view, short grains, int areasum, double px, double py, double sx, double sy, double sigma, double view_cx, double view_cy)
            {
                if (Total == 0)
                {
                    Rect.MinX = Rect.MaxX = px;
                    Rect.MinY = Rect.MaxY = py;
                }
                else
                {
                    if (Rect.MinX > px) Rect.MinX = px;
                    else if (Rect.MaxX < px) Rect.MaxX = px;
                    if (Rect.MinY > py) Rect.MinY = py;
                    else if (Rect.MaxY < py) Rect.MaxY = py;
                }
                Total++;
                if (side == 1) Side1Total++;
                else Side2Total++;
                bW.Write(zone);
                bW.Write(side);
                bW.Write(id);
                bW.Write(grains);
                bW.Write(areasum);
                bW.Write(px);
                bW.Write(py);
                bW.Write(sx);
                bW.Write(sy);
                bW.Write(sigma);
                bW.Write(id_view);
                bW.Write(view_cx);
                bW.Write(view_cy);
                IsUpdated = false;
            }

            internal void UpdateReader()
            {
                bW.Flush();
                bW.Seek(0, System.IO.SeekOrigin.Begin);
                bW.Write(Total);
                bW.Write(Rect.MinX);
                bW.Write(Rect.MaxX);
                bW.Write(Rect.MinY);
                bW.Write(Rect.MaxY);
                bW.Flush();
                bW.Seek(0, System.IO.SeekOrigin.End);
                IsUpdated = true;
            }

            internal void Link(string config)
            {
                if (!IsUpdated) UpdateReader();
                OpEmuRecHelper.LinkerExe.UseFileBacker = true;
                OpEmuRecHelper.LinkerExe.ProcessData(ReaderFile, TLGFile, config, null);
                System.IO.FileStream str = new System.IO.FileStream(TLGFile, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                SlopeCorr = new SySal.Scanning.PostProcessing.SlopeCorrections(str);
                str.Close();
            }

            internal void AddLayer(int id)
            {
                System.IO.FileStream f = null;
                try
                {
                    SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = new SySal.Scanning.Plate.IO.OPERA.LinkedZone((f = new System.IO.FileStream(TLGFile, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read)));
                    SySal.BasicTypes.Vector refcenter = new SySal.BasicTypes.Vector();
                    refcenter.X = (lz.Extents.MaxX + lz.Extents.MinX) * 0.5;
                    refcenter.Y = (lz.Extents.MaxY + lz.Extents.MinY) * 0.5;
                    refcenter.Z = Z;
                    System.Collections.ArrayList ai = new System.Collections.ArrayList(lz.Length);
                    int i;
                    for (i = 0; i < lz.Length; i++)
                        if (OpEmuRecHelper.IgnoreQ(lz[i]))
                            ai.Add(i);
                    SySal.TotalScan.Segment[] segs = new SySal.TotalScan.Segment[lz.Length];
                    for (i = 0; i < lz.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = lz[i].Info;
                        double dz = Z - info.Intercept.Z;
                        info.Intercept.Z = Z;
                        info.TopZ += dz;
                        info.BottomZ += dz;
                        segs[i] = new SySal.TotalScan.Segment(info, new SySal.TotalScan.BaseTrackIndex(i));
                    }
                    SySal.TotalScan.Layer tmpLayer = new SySal.TotalScan.Layer(id, SySal.OpEmuRecHelper.BrickId, this.PlateId, 0, refcenter, Z + 45.0, Z - 255.0);
                    if (segs.Length > 0) tmpLayer.AddSegments(segs);
                    OpEmuRecHelper.AORec.AddLayer(tmpLayer);
                    tmpLayer = null;
                    OpEmuRecHelper.AORec.SetAlignmentIgnoreList(id, (int [])ai.ToArray(typeof(int)));
                }
                finally
                {
                    if (f != null)
                    {
                        f.Close();
                        f = null;
                    }
                }
            }
        }

        static PlateInfo LastPlateAccessed = null;

        static System.Collections.ArrayList PlateInfoList = new System.Collections.ArrayList();

        static void AOReport(string text)
        {
            Console.WriteLine(text);
        }

        public static void Open(long brickid)
        {
            if (Uid != null) Close();
            ResetIterators();
            Vol = null;
            ViewMaps = null;
            BrickId = brickid;
            Uid = System.Guid.NewGuid().ToString();
            ScratchPath = System.Environment.GetEnvironmentVariable("SYSAL_SCRATCH_PATH");
            if (ScratchPath.EndsWith("/") == false && ScratchPath.EndsWith("\\") == false) ScratchPath += "/";
            ScratchPath += "oerhlp_" + Uid + "_" + brickid;
            PlateInfoList = new System.Collections.ArrayList();
            if (LinkerExe == null)
            {
                LinkerExe = new SySal.Executables.BatchLink.Exe();
                LinkerExe.UseFileBacker = true;
            }
            if (AORec == null)
            {
                AORec = new SySal.Processing.AlphaOmegaReconstruction.AlphaOmegaReconstructor();
                AORec.Report = new SySal.TotalScan.dReport(AOReport);
            }
            AORec.Clear();
        }

        public static void Close()
        {
            Vol = null;
            ViewMaps = null;
            ResetIterators();
            foreach (PlateInfo pl in PlateInfoList)
                pl.Dispose();
            PlateInfoList.Clear();
            LastPlateAccessed = null;
            Uid = null;
        }

        public static void AddPlate(int plate, double z)
        {
            foreach (PlateInfo pi in PlateInfoList)
                if (pi.PlateId == plate) throw new Exception("Plate " + plate + " already exists.");
            PlateInfoList.Add(new PlateInfo(plate, z));            
        }

        public static void AddMicrotrack(int plate, long zone, short side, int id, int id_view, short grains, int areasum, double px, double py, double sx, double sy, double sigma, double view_cx, double view_cy)
        {            
            if (LastPlateAccessed == null || LastPlateAccessed.PlateId != plate)
            {
                PlateInfo pa = null;
                foreach (PlateInfo pi in PlateInfoList)
                    if (pi.PlateId == plate)
                    {
                        pa = pi;
                        break;
                    }
                if (pa == null) throw new Exception("Plate " + plate + " does not exist.");
                LastPlateAccessed = pa;
            }
            LastPlateAccessed.AddMicrotrack(zone, side, id, id_view, grains, areasum, px, py, sx, sy, sigma, view_cx, view_cy);
        }

        public static void AlignAndFilter(string linkconfig, string alignconfig)
        {
            AORec.Clear();
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.AlphaOmegaReconstruction.Configuration));
            SySal.Processing.AlphaOmegaReconstruction.Configuration aorecconfig = (SySal.Processing.AlphaOmegaReconstruction.Configuration)xmls.Deserialize(new System.IO.StringReader(alignconfig));
            aorecconfig.MaxMissingSegments = 0;
            aorecconfig.VtxAlgorithm = SySal.Processing.AlphaOmegaReconstruction.VertexAlgorithm.None;
            AORec.Config = aorecconfig;
            double lastz = ((PlateInfo)(PlateInfoList[0])).Z;
            int i, layerid;
            for (i = 1; i < PlateInfoList.Count; i++)            
                if (((PlateInfo)(PlateInfoList[i])).Z > lastz)
                    lastz = ((PlateInfo)(PlateInfoList[i])).Z;
            lastz = lastz + 1.0;
            double nextz = 0.0;
            layerid = 0;
            while (true)
            {
                PlateInfo pa = null;
                for (i = 0; i < PlateInfoList.Count; i++)
                    if (((PlateInfo)(PlateInfoList[i])).Z < lastz)
                        if (pa == null || ((PlateInfo)(PlateInfoList[i])).Z > nextz)
                        {
                            pa = (PlateInfo)PlateInfoList[i];
                            nextz = pa.Z;
                        }
                if (pa == null) break;
                pa.Link(linkconfig);
                pa.AddLayer(layerid++);
                lastz = nextz;
            }
            string volpath = Uid + ".tsr";
            Vol = AORec.Reconstruct();
            SySal.Executables.TSRMuExpand.TLGspec[] tlgs = new SySal.Executables.TSRMuExpand.TLGspec[Vol.Layers.Length];
            for (i = 0; i < tlgs.Length; i++)
            {
                PlateInfo pa = (PlateInfo)PlateInfoList[i];
                tlgs[i].BrickId = BrickId;
                tlgs[i].SheetId = pa.PlateId;
                tlgs[i].Path = pa.TLGFile;                
            }
            Vol = SySal.Executables.TSRMuExpand.Exe.ProcessData(Vol, tlgs);
            ViewMaps = new ViewMap [Vol.Layers.Length][];
            for (layerid = 0; layerid < Vol.Layers.Length; layerid++)
            {
                SySal.TotalScan.Layer lay = Vol.Layers[layerid];
                System.Collections.ArrayList vm = new System.Collections.ArrayList();
                for (i = 0; i < lay.Length; i++)
                {
                    SySal.TotalScan.Segment seg = lay[i];
                    ViewMap nv = new ViewMap(seg);
                    int ifs = vm.BinarySearch(nv, nv);
                    if (ifs < 0) vm.Insert(~ifs, nv);
                    else ((ViewMap)vm[ifs]).Segments.Add(seg);
                    
                }
                for (i = 0; i < vm.Count; i++)
                    ((ViewMap)vm[i]).Segments.TrimToSize();
                ViewMaps[layerid] = (ViewMap[])vm.ToArray(typeof(ViewMap));
            }
            ResetIterators();
        }

        public static void Track(string linkconfig, string alignconfig)
        {            
            AORec.Clear();
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.AlphaOmegaReconstruction.Configuration));
            SySal.Processing.AlphaOmegaReconstruction.Configuration aorecconfig = (SySal.Processing.AlphaOmegaReconstruction.Configuration)xmls.Deserialize(new System.IO.StringReader(alignconfig));
            aorecconfig.MaxIters = 1;
            aorecconfig.VtxAlgorithm = SySal.Processing.AlphaOmegaReconstruction.VertexAlgorithm.None;
            AORec.Config = aorecconfig;
            double lastz = ((PlateInfo)(PlateInfoList[0])).Z;
            int i, layerid;
            for (i = 1; i < PlateInfoList.Count; i++)
                if (((PlateInfo)(PlateInfoList[i])).Z > lastz)
                    lastz = ((PlateInfo)(PlateInfoList[i])).Z;
            lastz = lastz + 1.0;
            double nextz = 0.0;
            layerid = 0;
            while (true)
            {
                PlateInfo pa = null;
                for (i = 0; i < PlateInfoList.Count; i++)
                    if (((PlateInfo)(PlateInfoList[i])).Z < lastz)
                        if (pa == null || ((PlateInfo)(PlateInfoList[i])).Z > nextz)
                        {
                            pa = (PlateInfo)PlateInfoList[i];
                            nextz = pa.Z;
                        }
                if (pa == null) break;
                pa.Link(linkconfig);
                pa.AddLayer(layerid++);
                lastz = nextz;
            }
            string volpath = Uid + ".tsr";
            Vol = AORec.Reconstruct();
            SySal.Executables.TSRMuExpand.TLGspec[] tlgs = new SySal.Executables.TSRMuExpand.TLGspec[Vol.Layers.Length];
            for (i = 0; i < tlgs.Length; i++)
            {
                PlateInfo pa = (PlateInfo)PlateInfoList[i];
                tlgs[i].BrickId = BrickId;
                tlgs[i].SheetId = pa.PlateId;
                tlgs[i].Path = pa.TLGFile;
            }
/*            Console.WriteLine("Written " + volpath);
            System.IO.FileStream sws = new System.IO.FileStream(volpath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
            Vol.Save(sws);
            sws.Flush();
            sws.Close();*/
            Vol = SySal.Executables.TSRMuExpand.Exe.ProcessData(Vol, tlgs);
/*            sws = new System.IO.FileStream("Q_" + volpath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
            Vol.Save(sws);
            sws.Flush();
            sws.Close();*/
            ViewMaps = new ViewMap[Vol.Layers.Length][];
            for (layerid = 0; layerid < Vol.Layers.Length; layerid++)
            {
                SySal.TotalScan.Layer lay = Vol.Layers[layerid];
                System.Collections.ArrayList vm = new System.Collections.ArrayList();
                for (i = 0; i < lay.Length; i++)
                {
                    SySal.TotalScan.Segment seg = lay[i];
                    ViewMap nv = new ViewMap(seg);
                    int ifs = vm.BinarySearch(nv, nv);
                    if (ifs < 0) vm.Insert(~ifs, nv);
                    else ((ViewMap)vm[ifs]).Segments.Add(seg);

                }
                for (i = 0; i < vm.Count; i++)
                    ((ViewMap)vm[i]).Segments.TrimToSize();
                ViewMaps[layerid] = (ViewMap[])vm.ToArray(typeof(ViewMap));
            }
            ResetIterators();
        }

        class ViewMap : System.Collections.IComparer
        {
            public long ZoneId;

            public System.Collections.ArrayList Segments;

            public ViewMap(SySal.TotalScan.Segment seg)
            {
                ZoneId = ((SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex)seg.Index).ZoneId;
                Segments = new System.Collections.ArrayList();
                Segments.Add(seg);
            }

            #region IComparer Members

            public int Compare(object x, object y)
            {
                return Math.Sign(((ViewMap)x).ZoneId - ((ViewMap)y).ZoneId);
            }

            #endregion
        }

        static SySal.TotalScan.Volume Vol = null;

        static ViewMap IterViewMap = null;

        static ViewMap[][] ViewMaps = null;

        static int IterIndex = -1;

        static int IterViewIndex = -1;

        static int IterLayerIndex = -1;

        static int IterTrackIndex = -1;

        static int IterVertexIndex = -1;

        static SySal.TotalScan.Track IterTrack = null;

        static SySal.TotalScan.Vertex IterVertex = null;

        static void ResetIterators()
        {
            IterViewMap = null;
            IterLayer = null;
            IterIndex = -1;
            IterViewIndex = -1;
            IterLayerIndex = -1;
            IterTrackIndex = -1;
            IterVertexIndex = -1;
            int mutks = 0;
            if (ViewMaps != null)
                foreach (ViewMap[] vma in ViewMaps)
                    if (vma != null)
                    {
                        foreach (ViewMap vm in vma)
                            mutks += vm.Segments.Count;
                    }                   
        }

        static PlateInfo IterPlateInfo = null;

        static bool IteratePlateInfo()
        {
            double lastz;
            int i;
            if (IterPlateInfo == null)
            {                
                lastz = ((PlateInfo)PlateInfoList[0]).Z;
                for (i = 1; i < PlateInfoList.Count; i++)
                    if (lastz < ((PlateInfo)PlateInfoList[i]).Z) lastz = ((PlateInfo)PlateInfoList[i]).Z;
                lastz += 1.0;
            }
            else lastz = IterPlateInfo.Z;            
            IterPlateInfo = null;
            foreach (PlateInfo pl in PlateInfoList)
            {
                if (pl.Z < lastz)
                    if (IterPlateInfo == null || pl.Z > IterPlateInfo.Z)
                        IterPlateInfo = pl;
            }            
            if (IterPlateInfo != null)
            {
                iPlateId = IterPlateInfo.PlateId;
                iPlateZ = IterPlateInfo.Z;
                iPlateSide1Total = IterPlateInfo.Side1Total;
                iPlateSide2Total = IterPlateInfo.Side2Total;
                return true;
            }
            return false;
        }

        static public int iPlateId;

        static public double iPlateZ;

        static public int iPlateSide1Total;

        static public int iPlateSide2Total;

        static SySal.TotalScan.Layer IterLayer = null;

        public static bool IterateLayer()
        {
            if (++IterLayerIndex >= Vol.Layers.Length) return false;
            IterLayer = Vol.Layers[IterLayerIndex];
            iLyPlateID = IterLayer.SheetId;
            iLySide = IterLayer.Side;
            SySal.Scanning.PostProcessing.SlopeCorrections slopecorr = null;
            foreach (PlateInfo pi in PlateInfoList)
                if (pi.PlateId == iLyPlateID)
                {
                    slopecorr = pi.SlopeCorr;
                    break;
                }
            SySal.BasicTypes.Vector r = IterLayer.RefCenter;
            iLyRefX = r.X;
            iLyRefY = r.Y;
            iLyRefZ = r.Z;
            iLyDownZ = IterLayer.DownstreamZ;
            iLyUpZ = IterLayer.UpstreamZ;
            SySal.TotalScan.AlignmentData a = IterLayer.AlignData;
            iLyAlignXX = a.AffineMatrixXX;
            iLyAlignXY = a.AffineMatrixXY;
            iLyAlignYX = a.AffineMatrixYX;
            iLyAlignYY = a.AffineMatrixYY;
            iLyAlignDX = a.TranslationX;
            iLyAlignDY = a.TranslationY;
            iLyAlignDZ = a.TranslationZ;
            if (iLySide == 1)
            {
                iLySlopeXX = iLyAlignXX * slopecorr.TopSlopeMultipliers.X;
                iLySlopeXY = iLyAlignXY * slopecorr.TopSlopeMultipliers.Y;
                iLySlopeYX = iLyAlignYX * slopecorr.TopSlopeMultipliers.X;
                iLySlopeYY = iLyAlignYY * slopecorr.TopSlopeMultipliers.Y;
                iLySlopeDX = iLyAlignXX * slopecorr.TopDeltaSlope.X + iLyAlignXY * slopecorr.TopDeltaSlope.Y;
                iLySlopeDY = iLyAlignYX * slopecorr.TopDeltaSlope.Y + iLyAlignYY * slopecorr.TopDeltaSlope.Y;
            }
            else
            {
                iLySlopeXX = iLyAlignXX * slopecorr.BottomSlopeMultipliers.X;
                iLySlopeXY = iLyAlignXY * slopecorr.BottomSlopeMultipliers.Y;
                iLySlopeYX = iLyAlignYX * slopecorr.BottomSlopeMultipliers.X;
                iLySlopeYY = iLyAlignYY * slopecorr.BottomSlopeMultipliers.Y;
                iLySlopeDX = iLyAlignXX * slopecorr.BottomDeltaSlope.X + iLyAlignXY * slopecorr.BottomDeltaSlope.Y;
                iLySlopeDY = iLyAlignYX * slopecorr.BottomDeltaSlope.Y + iLyAlignYY * slopecorr.BottomDeltaSlope.Y;
            }
            SySal.BasicTypes.Cuboid ext = Vol.Extents;
            iLyMinX = ext.MinX;
            iLyMaxX = ext.MaxX;
            iLyMinY = ext.MinY;
            iLyMaxY = ext.MaxY;
            return true;
        }

        public static int iLyPlateID = 0;

        public static int iLySide = 0;

        public static double iLyRefX = 0.0;

        public static double iLyRefY = 0.0;

        public static double iLyRefZ = 0.0;

        public static double iLyMinX = 0.0;

        public static double iLyMaxX = 0.0;

        public static double iLyMinY = 0.0;

        public static double iLyMaxY = 0.0;

        public static double iLyDownZ = 0.0;

        public static double iLyUpZ = 0.0;

        public static double iLyAlignXX = 0.0;

        public static double iLyAlignXY = 0.0;

        public static double iLyAlignYX = 0.0;

        public static double iLyAlignYY = 0.0;

        public static double iLyAlignDX = 0.0;

        public static double iLyAlignDY = 0.0;

        public static double iLyAlignDZ = 0.0;

        public static double iLySlopeXX = 1.0;

        public static double iLySlopeXY = 0.0;

        public static double iLySlopeYX = 0.0;

        public static double iLySlopeYY = 0.0;

        public static double iLySlopeDX = 0.0;

        public static double iLySlopeDY = 0.0;

        public static bool IterateTrack()
        {
            if (++IterTrackIndex >= Vol.Tracks.Length) return false;
            IterTrack = Vol.Tracks[IterTrackIndex];
            return true;
        }

        public static bool GetTrackSegment(int segid)
        {
            SySal.TotalScan.Segment seg = IterTrack[segid];
            SySal.Tracking.MIPEmulsionTrackInfo info = seg.Info;
            iGrains = info.Count;
            iAreaSum = (int)info.AreaSum;            
            iPosX = info.Intercept.X;
            iPosY = info.Intercept.Y;
            iPosZ = info.Intercept.Z;
            iSlopeX = info.Slope.X;
            iSlopeY = info.Slope.Y;
            iTopZ = info.TopZ;
            iBottomZ = info.BottomZ;
            iSigma = info.Sigma;
            try
            {
                SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex dbmi = (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex)seg.Index;
                iZoneId = dbmi.ZoneId;
                iSide = dbmi.Side;
                iId = dbmi.Id;
            }
            catch (Exception)
            {
                iZoneId = 0;
                iSide = seg.LayerOwner.Side;
                iId = 0;
            }
            iTrackId = IterTrack.Id;
            return true;
        }

        public static int GetTrackAttributes() { return iTkAttrList.Length; }

        public static string GetTrackAttributeName(int i) 
        {
            SySal.TotalScan.Attribute attr = iTkAttrList[i];
            try
            {
                return ((SySal.OperaDb.TotalScan.DBNamedAttributeIndex)attr.Index).Name; 
            }
            catch (Exception)
            {
                try
                {
                    return ((SySal.TotalScan.NamedAttributeIndex)attr.Index).Name;
                }
                catch (Exception)
                {
                    return "";
                }
            }
        }

        public static long GetTrackAttributeProcOp(int i)
        {
            SySal.TotalScan.Attribute attr = iTkAttrList[i];
            try
            {
                return ((SySal.OperaDb.TotalScan.DBNamedAttributeIndex)attr.Index).ProcOpId; 
            }
            catch (Exception)
            {
                return 0;
            }
        }

        public static double GetTrackAttribute(string name, long procop) 
        {
            try
            {
                if (procop == 0) return IterTrack.GetAttribute(new SySal.TotalScan.NamedAttributeIndex(name));
                else return IterTrack.GetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex(name, procop));                
            }
            catch (Exception)
            {
                return 0.0;
            }        
        }

        public static void SetTrackAttribute(string name, long procop, double v) 
        {
            IterTrack.SetAttribute(((procop == 0) ? (SySal.TotalScan.Index)new SySal.TotalScan.NamedAttributeIndex(name) : (SySal.TotalScan.Index)new SySal.OperaDb.TotalScan.DBNamedAttributeIndex(name, procop)), v); 
            iTkAttrList = IterTrack.ListAttributes();
            if (name.StartsWith(SySal.TotalScan.Flexi.Volume.DataSetString))
            {
                SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
                ds.DataType = name.Substring(SySal.TotalScan.Flexi.Volume.DataSetString.Length);
                ds.DataId = procop;
                ((SySal.TotalScan.Flexi.Track)IterTrack).DataSet = ds;
            }
        }

        public static int iTkCount = 0;

        public static double iTkDownX = 0;

        public static double iTkDownY = 0;

        public static double iTkDownZ = 0;

        public static double iTkDownSX = 0;

        public static double iTkDownSY = 0;

        public static int iTkDownVtxId = -1;

        public static double iTkDownIP = -1.0;

        public static double iTkUpX = 0;

        public static double iTkUpY = 0;

        public static double iTkUpZ = 0;

        public static double iTkUpSX = 0;

        public static double iTkUpSY = 0;

        public static int iTkUpVtxId = -1;

        public static double iTkUpIP = -1.0;

        public static SySal.TotalScan.Attribute[] iTkAttrList = null;

        public static bool IterateTrackKinematic()
        {
            if (++IterTrackIndex >= Vol.Tracks.Length) return false;
            IterTrack = Vol.Tracks[IterTrackIndex];
            iTkAttrList = IterTrack.ListAttributes();
            iTkCount = IterTrack.Length;
            iTkDownX = IterTrack.Downstream_PosX + (IterTrack.Downstream_Z - IterTrack.Downstream_PosZ) * IterTrack.Downstream_SlopeX;
            iTkDownY = IterTrack.Downstream_PosY + (IterTrack.Downstream_Z - IterTrack.Downstream_PosZ) * IterTrack.Downstream_SlopeY;
            iTkDownZ = IterTrack.Downstream_Z;
            iTkDownSX = IterTrack.Downstream_SlopeX;
            iTkDownSY = IterTrack.Downstream_SlopeY;
            if (IterTrack.Downstream_Vertex == null)
            {
                iTkDownVtxId = -1;
                iTkDownIP = -1.0;
            }
            else
            {
                iTkDownVtxId = IterTrack.Downstream_Vertex.Id;
                iTkDownIP = IterTrack.Downstream_Impact_Parameter;
            }
            iTkUpX = IterTrack.Upstream_PosX + (IterTrack.Upstream_Z - IterTrack.Upstream_PosZ) * IterTrack.Upstream_SlopeX;
            iTkUpY = IterTrack.Upstream_PosY + (IterTrack.Upstream_Z - IterTrack.Upstream_PosZ) * IterTrack.Upstream_SlopeY;
            iTkUpZ = IterTrack.Upstream_Z;
            iTkUpSX = IterTrack.Upstream_SlopeX;
            iTkUpSY = IterTrack.Upstream_SlopeY;
            if (IterTrack.Upstream_Vertex == null)
            {
                iTkUpVtxId = -1;
                iTkUpIP = -1.0;
            }
            else
            {
                iTkUpVtxId = IterTrack.Upstream_Vertex.Id;
                iTkUpIP = IterTrack.Upstream_Impact_Parameter;
            }
            return true;
        }

        public static int GetTrackLength()
        {
            return IterTrack.Length;
        }

        public static bool AccessTrackSegmentId(int segix)
        {
            SySal.TotalScan.Segment seg = null;
            try
            {
                seg = IterTrack[segix];
            }
            catch (Exception)
            {
                return false;
            }
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex dbmi = (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex)seg.Index;
            iZoneId = dbmi.ZoneId;
            iSide = dbmi.Side;
            iId = dbmi.Id;
            iTrackId = IterTrack.Id;
            return true;
        }

        public static bool IterateVertex()
        {
            if (++IterVertexIndex >= Vol.Vertices.Length) return false;
            IterVertex = Vol.Vertices[IterVertexIndex];
            iVtxAttrList = IterVertex.ListAttributes();
            iVtxID = IterVertex.Id;
            iVtxX = IterVertex.X;
            iVtxY = IterVertex.Y;
            iVtxZ = IterVertex.Z;
            iVtxAvgD = IterVertex.AverageDistance;
            IterVertexTrack = -1;
            return true;
        }

        public static int IterateVertexTrack()
        {
            if (++IterVertexTrack >= IterVertex.Length) return -1;
            return IterVertex[IterVertexTrack].Id;
        }

        public static int GetVertexAttributes() { return iVtxAttrList.Length; }

        public static string GetVertexAttributeName(int i)
        {
            SySal.TotalScan.Attribute attr = iVtxAttrList[i];
            try
            {
                return ((SySal.OperaDb.TotalScan.DBNamedAttributeIndex)attr.Index).Name;
            }
            catch (Exception)
            {
                try
                {
                    return ((SySal.TotalScan.NamedAttributeIndex)attr.Index).Name;
                }
                catch (Exception)
                {
                    return "";
                }
            }
        }

        public static long GetVertexAttributeProcOp(int i)
        {
            SySal.TotalScan.Attribute attr = iVtxAttrList[i];
            try
            {
                return ((SySal.OperaDb.TotalScan.DBNamedAttributeIndex)attr.Index).ProcOpId;
            }
            catch (Exception)
            {
                return 0;
            }
        }

        public static double GetVertexAttribute(string name, long procop)
        {
            try
            {
                if (procop == 0) return IterVertex.GetAttribute(new SySal.TotalScan.NamedAttributeIndex(name));
                else return IterVertex.GetAttribute(new SySal.OperaDb.TotalScan.DBNamedAttributeIndex(name, procop));
            }
            catch (Exception)
            {
                return 0.0;
            }
        }

        public static void SetVertexAttribute(string name, long procop, double v)
        {
            IterVertex.SetAttribute(((procop == 0) ? (SySal.TotalScan.Index)new SySal.TotalScan.NamedAttributeIndex(name) : (SySal.TotalScan.Index)new SySal.OperaDb.TotalScan.DBNamedAttributeIndex(name, procop)), v);
            iVtxAttrList = IterVertex.ListAttributes();
            if (name.StartsWith(SySal.TotalScan.Flexi.Volume.DataSetString))
            {
                SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
                ds.DataType = name.Substring(SySal.TotalScan.Flexi.Volume.DataSetString.Length);
                ds.DataId = procop;
                ((SySal.TotalScan.Flexi.Track)IterTrack).DataSet = ds;
            }
        }

        static int IterVertexTrack;

        public static int iVtxID;

        public static double iVtxX;

        public static double iVtxY;

        public static double iVtxZ;

        public static double iVtxAvgD;

        static SySal.TotalScan.Attribute[] iVtxAttrList = null;

        public static bool IterateMicrotrack()
        {
            SySal.TotalScan.Segment seg = null;
            if (IterViewMap == null || ++IterIndex >= IterViewMap.Segments.Count)            
            {
                if (IterViewMap == null)
                {
                    IterLayerIndex = 0;
                    IterViewIndex = -1;
                }
                IterViewMap = null;
                IterIndex = 0;
                while (ViewMaps.Length > IterLayerIndex)
                {
                    if (ViewMaps[IterLayerIndex].Length <= ++IterViewIndex)
                    {
                        IterViewIndex = -1;
                        IterLayerIndex++;
                    }
                    else
                    {
                        IterViewMap = ViewMaps[IterLayerIndex][IterViewIndex];
                        break;
                    }
                }
                if (IterViewMap == null) return false;
            }
            seg = (SySal.TotalScan.Segment)(IterViewMap.Segments[IterIndex]);

            SySal.Tracking.MIPEmulsionTrackInfo info = seg.Info;
            iGrains = info.Count;
            iAreaSum = (int)info.AreaSum;            
            iPosX = info.Intercept.X;
            iPosY = info.Intercept.Y;
            iPosZ = info.Intercept.Z;
            iSlopeX = info.Slope.X;
            iSlopeY = info.Slope.Y;
            iTopZ = info.TopZ;
            iBottomZ = info.BottomZ;
            iSigma = info.Sigma;
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex dbmi = (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex)seg.Index;
            iZoneId = dbmi.ZoneId;
            iSide = dbmi.Side;
            iId = dbmi.Id;
            iTrackId = (seg.TrackOwner == null) ? -1 : seg.TrackOwner.Id;
            return true;
        }

        public static bool TransformPoint(int plate, double x, double y, double z)
        {
            int i;
            for (i = 0; i < Vol.Layers.Length; i++)
            {
                if (Vol.Layers[i].SheetId == plate)
                {
                    SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
                    v.X = x;
                    v.Y = y;
                    v.Z = z;
                    v = Vol.Layers[i].ToAlignedPoint(v);
                    iTPointX = v.X;
                    iTPointY = v.Y;
                    iTPointZ = v.Z;
                }
            }
            return false;
        }

        public static int iGrains;
        public static int iAreaSum;
        public static double iPosX;
        public static double iPosY;
        public static double iPosZ;
        public static double iSlopeX;
        public static double iSlopeY;
        public static double iTopZ;
        public static double iBottomZ;
        public static double iSigma;
        public static long iZoneId;
        public static short iSide;
        public static int iId;
        public static int iTrackId;

        public static double iTPointX;
        public static double iTPointY;
        public static double iTPointZ;

        public static int GetTracks()
        {
            return Vol.Tracks.Length;
        }

        public static void TestLink(int plate, string configfile)
        {
            PlateInfo pa = null;
            foreach (PlateInfo pi in PlateInfoList)
                if (pi.PlateId == plate)
                {
                    pa = pi;
                    break;
                }
            if (pa == null) throw new Exception("Plate " + plate + " does not exist.");
            System.IO.StreamReader r = new System.IO.StreamReader(configfile);
            pa.Link(r.ReadToEnd());
            r.Close();
        }
    }
}
