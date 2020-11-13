using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.TSRMuExpand
{
    public struct TLGspec
    {
        public string Path;
        public long BrickId;
        public int SheetId;
        public TLGspec(long brickid, int sheetid, string path)
        {
            BrickId = brickid;
            SheetId = sheetid;
            Path = path;
        }
    }

    class TSR : SySal.TotalScan.Volume
    {
        #region Lists

        class LayerList : SySal.TotalScan.Volume.LayerList
        {
            public LayerList(SySal.TotalScan.Layer[] layers)
            {
                this.Items = layers;
            }
        }

        class TrackList : SySal.TotalScan.Volume.TrackList
        {
            public TrackList(SySal.TotalScan.Track[] tracks)
            {
                this.Items = tracks;
            }
        }

        class VertexList : SySal.TotalScan.Volume.VertexList
        {
            public VertexList(SySal.TotalScan.Vertex[] vertices)
            {
                this.Items = vertices;
            }
        }

        #endregion

        class USegment : SySal.TotalScan.Segment
        {
            public USegment(SySal.Tracking.MIPEmulsionTrackInfo info, int id, SySal.TotalScan.Index ix, SySal.TotalScan.Layer lay)
            {
                this.m_Info = info;
                this.m_LayerOwner = lay;
                this.m_PosInLayer = id;
                this.m_Index = ix;
            }
        }

        class Layer : SySal.TotalScan.Layer
        {
            internal int[] IdRemap;

            internal class DBMIReloc
            {
                public SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex Index;
                public int IdRemap;

                public DBMIReloc(SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex dbmi, int idremap)
                {
                    Index = dbmi;
                    IdRemap = idremap;
                }
            }

            internal class DBMIPMicroTrackIndexComparer : System.Collections.IComparer
            {
                public int Compare(object aa, object bb)
                {
                    DBMIReloc a = (DBMIReloc)aa;
                    DBMIReloc b = (DBMIReloc)bb;                    
                    if (a.Index.ZoneId < b.Index.ZoneId) return -1;
                    if (a.Index.ZoneId > b.Index.ZoneId) return 1;
                    if (a.Index.Side < b.Index.Side) return -1;
                    if (a.Index.Side > b.Index.Side) return -1;
                    return a.Index.Id - b.Index.Id;
                }

                internal static DBMIPMicroTrackIndexComparer TheComparer = new DBMIPMicroTrackIndexComparer();
            }            

            public Layer(int id, SySal.TotalScan.Layer inL, bool istop, SySal.Scanning.PostProcessing.SlopeCorrections sc, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, SySal.OperaDb.Scanning.DBMIPMicroTrackIndex dbmi)
            {                
                double x, y;
                this.m_Id = id;
                this.m_BrickId = inL.BrickId;
                this.m_SheetId = inL.SheetId;
                this.m_Side = (short)(istop ? 1 : 2);
                this.m_AlignmentData = inL.AlignData;
                this.m_RefCenter = inL.RefCenter;                
                double lzrefz;
                System.Collections.ArrayList dbmixs = new System.Collections.ArrayList();
                if (istop)
                {
                    lzrefz = lz.Top.BottomZ;
                    this.m_DownstreamZ = m_RefCenter.Z + sc.TopThickness;
                    this.m_UpstreamZ = m_RefCenter.Z;
                    this.m_AlignmentData.SAlignDSlopeX += this.m_AlignmentData.DShrinkX * sc.TopDeltaSlope.X;
                    this.m_AlignmentData.DShrinkX *= sc.TopSlopeMultipliers.X;
                    this.m_AlignmentData.SAlignDSlopeY += this.m_AlignmentData.DShrinkY * sc.TopDeltaSlope.Y;
                    this.m_AlignmentData.DShrinkY *= sc.TopSlopeMultipliers.Y;
                }
                else
                {
                    lzrefz = lz.Bottom.TopZ;
                    this.m_RefCenter.Z -= sc.BaseThickness;
                    this.m_DownstreamZ = m_RefCenter.Z;
                    this.m_UpstreamZ = m_DownstreamZ - sc.BottomThickness;
                    this.m_AlignmentData.SAlignDSlopeX += this.m_AlignmentData.DShrinkX * sc.BottomDeltaSlope.X;
                    this.m_AlignmentData.DShrinkX *= sc.BottomSlopeMultipliers.X;
                    this.m_AlignmentData.SAlignDSlopeY += this.m_AlignmentData.DShrinkY * sc.BottomDeltaSlope.Y;
                    this.m_AlignmentData.DShrinkY *= sc.BottomSlopeMultipliers.Y;
                }
                this.m_DownstreamZ_Updated = this.m_UpstreamZ_Updated = true;
                IdRemap = new int[inL.Length];
                System.Collections.ArrayList tSegs = new System.Collections.ArrayList();                
                int i, j;
                for (i = 0; i < inL.Length; i++)
                {                    
                    SySal.TotalScan.Segment s = inL[i];
                    j = ((SySal.TotalScan.BaseTrackIndex)s.Index).Id;
                    SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex mi = istop ? dbmi.TopTracksIndex[j] : dbmi.BottomTracksIndex[j];
                    if (mi.Id < 0)
                    {
                        IdRemap[i] = -1;
                        continue;
                    }
                    DBMIReloc ixr = new DBMIReloc(mi, -1);
                    int ixpos = dbmixs.BinarySearch(ixr, DBMIPMicroTrackIndexComparer.TheComparer);
                    if (ixpos < 0)
                    {
                        ixr.IdRemap = tSegs.Count;
                        dbmixs.Insert(~ixpos, ixr);
                    }
                    else
                    {
                        IdRemap[i] = ((DBMIReloc)dbmixs[ixpos]).IdRemap;
                        continue;
                    }
                    SySal.Tracking.MIPEmulsionTrackInfo info = istop ? lz[j].Top.Info : lz[j].Bottom.Info;
                    info.Intercept.X += info.Slope.X * (lzrefz - info.Intercept.Z);
                    info.Intercept.Y += info.Slope.Y * (lzrefz - info.Intercept.Z);
                    info.Intercept.Z = m_RefCenter.Z;
                    info.TopZ = info.TopZ - lzrefz + m_RefCenter.Z;
                    info.BottomZ = info.BottomZ - lzrefz + m_RefCenter.Z;
                    x = info.Intercept.X - m_RefCenter.X;
                    y = info.Intercept.Y - m_RefCenter.Y;
                    info.Intercept.X = m_AlignmentData.AffineMatrixXX * x + m_AlignmentData.AffineMatrixXY * y + m_RefCenter.X + m_AlignmentData.TranslationX;
                    info.Intercept.Y = m_AlignmentData.AffineMatrixYX * x + m_AlignmentData.AffineMatrixYY * y + m_RefCenter.Y + m_AlignmentData.TranslationY;
                    x = m_AlignmentData.DShrinkX * info.Slope.X + m_AlignmentData.SAlignDSlopeX;
                    y = m_AlignmentData.DShrinkY * info.Slope.Y + m_AlignmentData.SAlignDSlopeY;
                    info.Slope.X = m_AlignmentData.AffineMatrixXX * x + m_AlignmentData.AffineMatrixXY * y;
                    info.Slope.Y = m_AlignmentData.AffineMatrixYX * x + m_AlignmentData.AffineMatrixYY * y;
                    SySal.TotalScan.Segment ns = new USegment(info, tSegs.Count, mi, this);                    
                    IdRemap[i] = tSegs.Count;
                    tSegs.Add(ns);
                }
                this.Segments = (SySal.TotalScan.Segment[])tSegs.ToArray(typeof(SySal.TotalScan.Segment));
            }
        }

        void WriteDuplicatesHeader()
        {
            Console.WriteLine("Duplicates found:");
            Console.WriteLine("Track\tLength\tDVLen\tUVLen\tLayer\tSegment\tCnflct\tCnfLen\tCDVLen\tCUVLen\tIsTop");
        }

        void WriteDuplicate(SySal.TotalScan.Track tk, int lay, int seg, int side, SySal.TotalScan.Track conftk)
        {
            Console.WriteLine(tk.Id + "\t" + tk.Length + "\t" + (tk.Downstream_Vertex == null ? 0 : tk.Downstream_Vertex.Length) + "\t" + (tk.Upstream_Vertex == null ? 0 : tk.Upstream_Vertex.Length) + "\t" + lay + "\t" + seg +
                "\t" + conftk.Id + "\t" + conftk.Length + "\t" + (conftk.Downstream_Vertex == null ? 0 : conftk.Downstream_Vertex.Length) + "\t" + (conftk.Upstream_Vertex == null ? 0 : conftk.Upstream_Vertex.Length) + "\t" + side);
        }

        void WriteDuplicatesAdvice()
        {
            Console.WriteLine("This TSR file is not suitable for DB insertion.");
        }

        internal class UtilityTrack : SySal.TotalScan.Track
        {
            public static void PutId(SySal.TotalScan.Track t, int i)
            {
                SySal.TotalScan.Track.SetId(t, i);
            }
        }

        internal class UtilityVertex : SySal.TotalScan.Vertex
        {
            public static void PutId(SySal.TotalScan.Vertex v, int i)
            {
                SySal.TotalScan.Vertex.SetId(v, i);
            }
        }

        public TSR(SySal.TotalScan.Volume v, TLGspec[] tlgspecs)
        {
            bool DuplicatesFound = false;
            this.m_Extents = v.Extents;
            this.m_Id = v.Id;
            this.m_RefCenter = v.RefCenter;

            SySal.TotalScan.Layer[] tLayers = new SySal.TotalScan.Layer[2 * v.Layers.Length];
            SySal.TotalScan.Track[] tTracks = new SySal.TotalScan.Track[v.Tracks.Length];
            SySal.TotalScan.Vertex[] tVertices = new SySal.TotalScan.Vertex[v.Vertices.Length];

            int i;
            for (i = 0; i < v.Layers.Length; i++)
            {
                SySal.Scanning.PostProcessing.SlopeCorrections sc = (SySal.Scanning.PostProcessing.SlopeCorrections)SySal.OperaPersistence.Restore(tlgspecs[i].Path, typeof(SySal.Scanning.PostProcessing.SlopeCorrections));
                //SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(tlgspecs[i].Path, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
                SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = SySal.DataStreams.OPERALinkedZone.FromFile(tlgspecs[i].Path);
                SySal.OperaDb.Scanning.DBMIPMicroTrackIndex dbmi = (SySal.OperaDb.Scanning.DBMIPMicroTrackIndex)SySal.OperaPersistence.Restore(tlgspecs[i].Path, typeof(SySal.OperaDb.Scanning.DBMIPMicroTrackIndex));
                tLayers[i * 2] = new Layer(i * 2, v.Layers[i], true, sc, lz, dbmi);
                tLayers[i * 2 + 1] = new Layer(i * 2 + 1, v.Layers[i], false, sc, lz, dbmi);
            }

            this.m_Layers = new LayerList(tLayers);

            int j, ly, lp, np;
            Layer lay;
            for (i = 0; i < v.Tracks.Length; i++)
            {
                SySal.TotalScan.Track tk = v.Tracks[i];
                SySal.TotalScan.Track ntk = new SySal.TotalScan.Track(i);
                for (j = 0; j < tk.Length; j++)
                {
                    SySal.TotalScan.Segment s = tk[j];
                    ly = s.LayerOwner.Id;
                    lp = s.PosInLayer;
                    lay = (Layer)tLayers[ly * 2];
                    if ((np = lay.IdRemap[lp]) >= 0)
                        if (lay[np].TrackOwner != null) 
                        {
                            if (!DuplicatesFound)
                            {
                                DuplicatesFound = true;
                                WriteDuplicatesHeader();
                            }
                            WriteDuplicate(tk, s.LayerOwner.Id, s.PosInLayer, 1, v.Tracks[lay[np].TrackOwner.Id]);
                            if (tk.Length > v.Tracks[ntk.Id].Length)
                               lay.AddSegment(lay[np]); // throw new Exception("Two base tracks found sharing the same microtrack! Expansion cannot continue.");
                        }
                        else ntk.AddSegment(lay[np]);                    
                    lay = (Layer)tLayers[ly * 2 + 1];
                    if ((np = lay.IdRemap[lp]) >= 0)
                        if (lay[np].TrackOwner != null)
                        {
                            if (!DuplicatesFound)
                            {
                                DuplicatesFound = true;
                                WriteDuplicatesHeader();
                            }
                            WriteDuplicate(tk, s.LayerOwner.Id, s.PosInLayer, 2, v.Tracks[lay[np].TrackOwner.Id]);
                            if (tk.Length > v.Tracks[ntk.Id].Length)
                                lay.AddSegment(lay[np]); // throw new Exception("Two base tracks found sharing the same microtrack! Expansion cannot continue.");
                        }
                        else ntk.AddSegment(lay[np]);                    
                }
                SySal.TotalScan.Attribute[] attr = tk.ListAttributes();
                foreach (SySal.TotalScan.Attribute a in attr)
                    ntk.SetAttribute(a.Index, a.Value);
                tTracks[i] = ntk;
            }
            int[] idmap = new int[tTracks.Length];
            for (i = j = 0; i < tTracks.Length; i++)
                if (tTracks[i].Length > 0) idmap[i] = j++;
                else idmap[i] = -1;
            if (j < i)
            {
                SySal.TotalScan.Track[] otTracks = tTracks;
                tTracks = new SySal.TotalScan.Track[j];
                for (i = j = 0; i < otTracks.Length; i++)
                    if (otTracks[i].Length > 0)
                    {
                        tTracks[j] = otTracks[i];
                        UtilityTrack.PutId(tTracks[j], j);
                        j++;
                    }
            }

            this.m_Tracks = new TrackList(tTracks);

            for (i = 0; i < v.Vertices.Length; i++)
            {
                SySal.TotalScan.Vertex vtx = v.Vertices[i];
                SySal.TotalScan.Vertex nvtx = new SySal.TotalScan.Vertex(i);
                for (j = 0; j < vtx.Length; j++)
                {
                    SySal.TotalScan.Track tk = vtx[j];
                    if (idmap[tk.Id] >= 0)
                    {
                        SySal.TotalScan.Track ntk = tTracks[idmap[tk.Id]];
                        if (tk.Upstream_Vertex == vtx)
                        {
                            nvtx.AddTrack(ntk, true);
                            ntk.SetUpstreamVertex(nvtx);
                        }
                        else
                        {
                            nvtx.AddTrack(ntk, false);
                            ntk.SetDownstreamVertex(nvtx);
                        }
                    }
                }
                SySal.TotalScan.Attribute[] attr = vtx.ListAttributes();
                foreach (SySal.TotalScan.Attribute a in attr)
                    nvtx.SetAttribute(a.Index, a.Value);
                tVertices[i] = nvtx;
            }
            for (i = j = 0; i < tVertices.Length; i++)
                if (tVertices[i].Length >= 2) j++;
            if (j < i)
            {
                SySal.TotalScan.Vertex[] otVertices = tVertices;
                tVertices = new SySal.TotalScan.Vertex[j];
                for (i = j = 0; i < otVertices.Length; i++)
                    if (otVertices[i].Length >= 2)
                    {
                        tVertices[j] = otVertices[i];
                        UtilityVertex.PutId(tVertices[j], j);
                        j++;
                    }
            }

            this.m_Vertices = new VertexList(tVertices);

            //if (DuplicatesFound) WriteDuplicatesAdvice();
        }
    }

    /// <summary>
    /// TSRMuExpand - command line tool to transform basetrack-based TSR files to microtrack-based TSR files, ready for insertion into DB.
    /// </summary>
    /// <remarks>
    /// <para>In common use, TSR files are generated starting from sets of basetracks (and promoted microtracks). In order for these TSR files to be inserted
    /// into the DB, they must be converted to files based on microtracks. Since TSRMuExpand does not perform any new reconstruction, the information to
    /// work back the path to original microtracks must be present in source TLGs.</para>
    /// <para>TSRMuExpand expects the following command line:</para>
    /// <para><example><code>TSRMuExpand.exe &lt;input TSR file path&gt; &lt;TLG list file&gt; &lt;output Opera persistence path for TSR&gt;</code></example></para>
    /// <para>Notice the input must be a file, whereas the output can be a file as well as a DB persistence path.</para>
    /// <para>The TLG list file allows TSRMuExpand to trace back, for each layer, the indices of tracks and their associated microtrack information. The file 
    /// should be formatted as a sequence of lines, each line with the format:
    /// <c>BrickId SheetId TLGpath</c> where <c>BrickId</c> is the ID of the brick, <c>SheetId</c> is the Id number of the plate, and <c>TLGpath</c> is the path to the TLG file used as the input for reconstruction
    /// on that plate.</para>
    /// <para>TLG files are valid for TSRMuExpand if they are MultiSection TLGs containing:
    /// <list type="bullet">
    /// <item>A <see cref="SySal.OperaDb.Scanning.DBMIPMicroTrackIndex"/> section with the indices of original microtracks.</item>
    /// <item>A <see cref="SySal.Scanning.PostProcessing.SlopeCorrections"/> section with the measured thickness of emulsion layers and base, and the adjusted values of 
    /// slope correction parameters.</item>
    /// </list></para>
    /// All other sections are optional and are ignored by TSRMuExpand.
    /// </remarks>
    public class Exe
    {
        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("usage: TSRMuExpand.exe <input TSR file path> <TLG list file> <output Opera persistence path for TSR>");
                Console.WriteLine("TLG list file format:");
                Console.WriteLine("BrickId SheetId TLGpath (one triple per line)");
                Console.WriteLine("All sheet ids present in the TSR need to be specified.");
                return;
            }
            System.Collections.ArrayList sheetspecs = new System.Collections.ArrayList();
            System.IO.StreamReader r = new System.IO.StreamReader(args[1]);
            string line;
            while ((line = r.ReadLine()) != null)
            {
                line = line.Trim();
                string[] token = line.Split(' ', '\t');
                if (token.Length != 3) continue;
                TLGspec newspec = new TLGspec(Convert.ToInt64(token[0]), Convert.ToInt32(token[1]), token[2]);
                foreach (TLGspec o in sheetspecs)
                    if (o.SheetId == newspec.SheetId && o.BrickId == newspec.BrickId)
                        throw new Exception("Brick id " + newspec.BrickId + " Sheet id " + newspec.SheetId + " has been specified twice! Aborting.");
                sheetspecs.Add(newspec);
            }
            SySal.TotalScan.BaseTrackIndex.RegisterFactory();
            SySal.TotalScan.NamedAttributeIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBNamedAttributeIndex.RegisterFactory();
            SySal.TotalScan.Volume inV = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(args[0], typeof(SySal.TotalScan.Volume));            
            Console.WriteLine("Written " + SySal.OperaPersistence.Persist(args[2], ProcessData(inV, (TLGspec [])sheetspecs.ToArray(typeof(TLGspec)))));
        }

        public static SySal.TotalScan.Volume ProcessData(SySal.TotalScan.Volume inV, TLGspec[] sheetspecs)
        {
            TLGspec[] layerTLG = new TLGspec[inV.Layers.Length];
            int i;
            for (i = 0; i < inV.Layers.Length; i++)
            {
                layerTLG[i].Path = null;
                foreach (TLGspec o in sheetspecs)
                    if (o.BrickId == inV.Layers[i].BrickId && o.SheetId == inV.Layers[i].SheetId)
                    {
                        layerTLG[i] = o;
                        break;
                    }
                if (layerTLG[i].Path == null)
                    throw new Exception("Layer " + i + " Brick id " + inV.Layers[i].BrickId + " Sheet id " + inV.Layers[i].SheetId + " has no mapping TLG! Aborting.");
            }
            return new TSR(inV, layerTLG);
        }
    }
}
