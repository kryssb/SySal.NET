using System;
using SySal;
using SySal.Management;
using SySal.Imaging;
using SySal.Tracking;
using SySal.Scanning;
using SySal.Scanning.Plate;
using SySal.Scanning.PostProcessing;
using SySal.Scanning.PostProcessing.FragmentLinking;
using System.Windows.Forms;
using System.Collections;
using System.Runtime.Serialization;
using SySal.Scanning.Plate.IO.OPERA.RawData;
using System.Xml.Serialization;

namespace SySal.Processing.StripesFragLink2
{
    /// <summary>
    /// Configuration for StripesFragmentLinker.
    /// </summary>
    [Serializable]
    [XmlType("StripesFragLink2.Configuration")]
    public class Configuration : SySal.Management.Configuration
    {
        /// <summary>
        /// Minimum number of grains to link a microtrack.
        /// </summary>
        public int MinGrains;
        /// <summary>
        /// Minimum slope to link a microtrack (useful to avoid camera spots artifacts).
        /// </summary>
        public double MinSlope;
        /// <summary>
        /// Position tolerance to merge two microtracks (this is part of 
        /// the double-reconstruction cleaning, especially in the overlap region between 
        /// adjacent fields of view).
        /// </summary>
        public double MergePosTol;
        /// <summary>
        /// Slope tolerance to merge two microtracks (this is part of 
        /// the double-reconstruction cleaning, especially in the overlap region between 
        /// adjacent fields of view).
        /// </summary>
        public double MergeSlopeTol;
        /// <summary>
        /// Position tolerance for microtrack linking. Two microtracks can be linked if the 
        /// extrapolation on either microtrack falls near the other within this tolerance.
        /// </summary>
        public double PosTol;
        /// <summary>
        /// Slope tolerance for microtrack linking. Two microtracks can be linked if slopes
        /// are closer than this tolerance in the transverse direction of the possibel base-track.		
        /// </summary>
        public double SlopeTol;
        /// <summary>
        /// Slope tolerance for microtrack linking. Two microtracks can be linked if slopes 
        /// are closer than the tolerance expressed by the formula <c>SlopeTol + Slope * SlopeTolIncreaseWithSlope</c> 
        /// in the longitudinal direction of the possibel base-track.
        /// </summary>
        public double SlopeTolIncreaseWithSlope;
        /// <summary>
        /// Memory saving level. Usually also improves the speed (reduced off-cache page hits). 4 levels
        /// are supported:
        /// <list type="table">
        /// <listheader><term>Level</term><description>Behaviour</description></listheader>
        /// <item><term>0</term><description>No memory saving applied.</description></item>
        /// <itea><term>1</term><description>Save track data to temporary files.</description></itea>
        /// <item><term>2</term><description>Save track data to temporary files and swap them off to disk as soon as they've been used, even if it is not finished with them.</description></item>		
        /// <item><term>3</term><description>Maximum: keep track data to temporary files as long as possible.</description></item>
        /// </list>
        /// </summary>
        public uint MemorySaving;
        /// <summary>
        /// Disables multithreading if <c>true</c>. If <c>false</c>, one thread per processor is used.
        /// </summary>
        public bool SingleThread = false;
        /// <summary>
        /// If <c>true</c> the same microtrack cannot be used twice. Duplicates are allowed if set to <c>false</c>.
        /// </summary>
        public bool PreventDuplication;
        /// <summary>
        /// If true, only microtracks linked in a base-track are kept; the others are discarded. 
        /// If false, all microtracks are kept.
        /// </summary>
        public bool KeepLinkedTracksOnly;
        /// <summary>
        /// If true, view information is preserved.
        /// </summary>
        public bool PreserveViews;
        /// <summary>
        /// Inline quality cut. If null or "", no quality cut is applied. If the quality cut string
        /// is non-null, it must contain at least one parameter. The known parameters are:
        /// <list type="table">
        /// <listheader><term>Name</term><description>Meaning</description></listheader>
        /// <item><term>A</term><description>AreaSum of the base-track</description></item>
        /// <item><term>TA</term><description>AreaSum of the top microtrack</description></item>
        /// <item><term>BA</term><description>AreaSum of the bottom microtrack</description></item>
        /// <item><term>N</term><description>Grains in the base-track</description></item>
        /// <item><term>TN</term><description>Grains in the top microtrack</description></item>
        /// <item><term>BN</term><description>Grains in the bottom microtrack</description></item>
        /// <item><term>PX,Y</term><description>X,Y position of the base-track at the top edge of the base.</description></item>
        /// <item><term>TPX,Y</term><description>X,Y position of the top microtrack at the top edge of the base.</description></item>
        /// <item><term>BPX,Y</term><description>X,Y position of the bottom microtrack at the bottom edge of the base.</description></item>
        /// <item><term>PZ</term><description>Z position of the base-track at the top edge of the base.</description></item>
        /// <item><term>TPZ</term><description>Z position of the top microtrack at the top edge of the base.</description></item>
        /// <item><term>BPZ</term><description>Z position of the bottom microtrack at the bottom edge of the base.</description></item>
        /// <item><term>SX,Y</term><description>X,Y slope of the base-track.</description></item>
        /// <item><term>TSX,Y</term><description>X,Y slope of the top microtrack.</description></item>
        /// <item><term>BSX,Y</term><description>X,Y slope of the bottom microtrack.</description></item>
        /// <item><term>S</term><description>Sigma of the base-track.</description></item>
        /// <item><term>TS</term><description>Sigma of the top microtrack.</description></item>
        /// <item><term>BS</term><description>Sigma of the bottom microtrack.</description></item>
        /// </list>
        /// </summary>
        public string QualityCut = "";
        /// <summary>
        /// Criterion to promote unlinked microtracks to <i>weak base tracks</i>. A weak base track is a base track made of only one microtrack. 
        /// The missing microtrack is simply the extrapolation of the existing microtrack, but with zero Grains and zero AreaSum. By definition, <c>Sigma</c> is <c>negative</c> for weak base tracks. This implementation assigns <c>-1.0</c>.
        /// If <c>MicrotrackPromotion</c>is non-null, it must contain at least one parameter. The known parameters are:
        /// <list type="table">
        /// <listheader><term>Name</term><description>Meaning</description></listheader>
        /// <item><term>A</term><description>AreaSum of the microtrack</description></item>
        /// <item><term>N</term><description>Grains in the microtrack</description></item>
        /// <item><term>SX,Y</term><description>X,Y slope of the microtrack.</description></item>
        /// <item><term>S</term><description>Sigma of the microtrack.</description></item>
        /// </list>
        /// </summary>
        public string MicrotrackPromotion = "";
        /// <summary>
        /// The maximum number of base tracks and/or promoted microtracks to be generated in a view. Used to avoid cluttered views near edges and spots. Set to <c>0</c> to remove any limitation.
        /// </summary>
        public int LinkLimit = 0;
        /// <summary>
        /// The path of the file where the link process is logged. If set to an empty string, no log is produced.
        /// </summary>
        public string DumpFilePath = "";
        /// <summary>
        /// Builds an empty configuration.
        /// </summary>
        public Configuration() : base("") { }
        /// <summary>
        /// Builds an empty configuration with a name.
        /// </summary>
        /// <param name="name">the name to be assigned to the configuration.</param>
        public Configuration(string name) : base(name) { }
        /// <summary>
        /// Clones the configuration.
        /// </summary>
        /// <returns>the object clone.</returns>
        public override object Clone()
        {
            Configuration c = new Configuration(Name);
            c.MinGrains = MinGrains;
            c.MinSlope = MinSlope;
            c.MergePosTol = MergePosTol;
            c.MergeSlopeTol = MergeSlopeTol;
            c.PosTol = PosTol;
            c.SlopeTol = SlopeTol;
            c.SlopeTolIncreaseWithSlope = SlopeTolIncreaseWithSlope;
            c.MemorySaving = MemorySaving;
            c.SingleThread = SingleThread;
            c.PreventDuplication = PreventDuplication;
            c.KeepLinkedTracksOnly = KeepLinkedTracksOnly;
            c.QualityCut = QualityCut;
            c.PreserveViews = PreserveViews;
            c.LinkLimit = LinkLimit;
            c.DumpFilePath = DumpFilePath;
            return c;
        }
    }


    /// <summary>
    /// Fragment linking class that works stripe by stripe along the X axis.
    /// </summary>
    /// <remarks>
    /// <para>This fragment linking algorithm tries to make minimum use of the machine RAM. 
    /// The limiting factor for this kind of application is indeed memory: when disk swapping 
    /// begins, the processing speed goes down by a factor 100 or more. Temporary files are
    /// generated in the user's <c>%TEMP%</c> directory. If the processing is interrupted,
    /// and finalizers are not called (e.g., interruption by TaskManager) the <c>%TEMP%</c>
    /// directory is left dirty and has to be cleaned manually.</para>
    /// <para>In order to minimize the amount of RAM needed, linking proceeds by horizontal 
    /// rows of views. At each moment, only two adjacent rows of views are present in memory, 
    /// all the others residing on the disk. When linking is complete, the row with the lower
    /// Y coordinate is discarded, and the next row (next higher Y) is loaded.</para>
    /// </remarks>
    [Serializable]
    [XmlType("StripesFragLink2.StripesFragmentLinker")]
    public class StripesFragmentLinker : IFragmentLinker, IManageable, ILinkProcessor
    {
        [NonSerialized]
        private NumericalTools.Function QF = null;

        [NonSerialized]
        private string[] QFParams = new string[0];

        internal delegate bool dQualityCut(IntMIPIndexedBaseTrack tk);

        [NonSerialized]
        private NumericalTools.Function PF = null;

        [NonSerialized]
        private string[] PFParams = new string[0];

        internal delegate bool dPromotion(IntTrack tk);

        [NonSerialized]
        private dPromotion PC;

        [NonSerialized]
        private StripesFragLink2.Configuration C;

        [NonSerialized]
        private string intName;

        [NonSerialized]
        private dShouldStop intShouldStop;

        [NonSerialized]
        private dProgress intProgress;

        [NonSerialized]
        private dLoadFragment intLoad;

        [NonSerialized]
        private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        #region Management

        public StripesFragmentLinker()
        {
            //
            // TODO: Add constructor logic here
            //
            C = new StripesFragLink2.Configuration("Default Stripes FragmentLinker2 Config");
            C.MinGrains = 6;
            C.PosTol = 50.0f;
            C.SlopeTol = 0.03f;
            C.SlopeTolIncreaseWithSlope = 0.2f;
            C.MergePosTol = 20.0f;
            C.MergeSlopeTol = 0.05f;
            C.MinSlope = 0.010f;
            C.MemorySaving = 2;
            C.KeepLinkedTracksOnly = true;
            C.PreserveViews = false;
            C.QualityCut = "";

            intName = "Default Stripes FragmentLinker2";            
            PC = new dPromotion(PCApply);
        }

        public string Name
        {
            get
            {
                return intName;
            }
            set
            {
                intName = value;
            }
        }

        [XmlElement(typeof(StripesFragLink2.Configuration))]
        public SySal.Management.Configuration Config
        {
            get
            {
                return C;
            }
            set
            {
                C = (StripesFragLink2.Configuration)value;
                QF = null;
                QFParams = new string[0];
                string qf = null;
                if (C.QualityCut == null || (qf = C.QualityCut.Trim()).Length == 0) QF = null;
                else
                {
                    try
                    {
                        QF = new NumericalTools.CStyleParsedFunction(qf);
                    }
                    catch (Exception)
                    {
                        try
                        {
                            QF = new NumericalTools.BASICStyleParsedFunction(qf);
                        }
                        catch (Exception x)
                        {
                            throw new Exception("Can't interpret expression:\r\n" + x.Message);
                        }
                    }
                    QFParams = QF.ParameterList;
                    if (QFParams == null || QFParams.Length == 0)
                    {
                        QFParams = new string[0];
                        throw new Exception("Quality cut must contain at least one track parameter!");
                    }
                    int i;
                    for (i = 0; i < QFParams.Length; i++)
                        QFParams[i] = QFParams[i].ToUpper();
                }
                PF = null;
                PFParams = new string[0];
                string pf = null;
                if (C.MicrotrackPromotion == null || (pf = C.MicrotrackPromotion.Trim()).Length == 0) PF = null;
                else
                {
                    try
                    {
                        PF = new NumericalTools.CStyleParsedFunction(pf);
                    }
                    catch (Exception)
                    {
                        try
                        {
                            PF = new NumericalTools.BASICStyleParsedFunction(pf);
                        }
                        catch (Exception x)
                        {
                            throw new Exception("Can't interpret expression:\r\n" + x.Message);
                        }
                    }
                    PFParams = PF.ParameterList;
                    if (PFParams == null || PFParams.Length == 0)
                    {
                        PFParams = new string[0];
                        throw new Exception("Microtrack promotion selection needs at least one parameter!");
                    }
                    int i;
                    for (i = 0; i < PFParams.Length; i++)
                        PFParams[i] = PFParams[i].ToUpper();
                }
            }
        }

        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            bool ret;
            EditConfigForm myform = new EditConfigForm();
            myform.C = (StripesFragLink2.Configuration)c.Clone();
            if ((ret = (myform.ShowDialog() == DialogResult.OK))) c = myform.C;
            myform.Dispose();
            return ret;
        }

        [XmlIgnore]
        public IConnectionList Connections
        {
            get
            {
                return EmptyConnectionList;
            }
        }

        [XmlIgnore]
        public bool MonitorEnabled
        {
            get
            {
                return false;
            }
            set
            {
                if (value != false) throw new System.Exception("This object has no monitor.");
            }
        }
        #endregion

        #region IFragmentLinker

        #region Internals

        internal class RefDepths
        {
            public double TopExt, TopInt, BottomInt, BottomExt;
            public int Hits;
        }

        private class LinkedZone : SySal.Scanning.Plate.LinkedZone
        {
            public override void Save(System.IO.Stream s)
            {
                throw new Exception("A generic LinkedZone cannot be saved.");
            }

            internal LinkedZone(Catalog Cat, IntMIPIndexedBaseTrack[] Linked, RefDepths rd)
                : base()
            {
                int i;
                m_Id = Cat.Id;
                m_Center.X = (Cat.Extents.MinX + Cat.Extents.MaxX) * 0.5f;
                m_Center.Y = (Cat.Extents.MinY + Cat.Extents.MaxY) * 0.5f;
                SySal.Scanning.MIPIndexedEmulsionTrack[] toptks = new SySal.Scanning.MIPIndexedEmulsionTrack[Linked.Length];
                SySal.Scanning.MIPIndexedEmulsionTrack[] bottomtks = new SySal.Scanning.MIPIndexedEmulsionTrack[Linked.Length];
                SySal.Scanning.MIPBaseTrack[] basetks = new SySal.Scanning.MIPBaseTrack[Linked.Length];
                for (i = 0; i < Linked.Length; i++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo tinfo = Linked[i].Top.Info;
                    double dz = (double)rd.TopInt - tinfo.Intercept.Z;
                    tinfo.Field = 0;
                    tinfo.Intercept.Z += dz;
                    tinfo.TopZ += dz;
                    tinfo.BottomZ += dz;
                    toptks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(tinfo, null, i);
                    SySal.Tracking.MIPEmulsionTrackInfo binfo = Linked[i].Bottom.Info;
                    binfo.Field = 0;
                    binfo.Intercept.Z += dz;
                    binfo.TopZ += dz;
                    binfo.BottomZ += dz;
                    bottomtks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(binfo, null, i);
                    SySal.Tracking.MIPEmulsionTrackInfo info = Linked[i].Info;
                    info.Field = 0;
                    info.Intercept.Z += dz;
                    info.TopZ += dz;
                    info.BottomZ += dz;
                    basetks[i] = new IntMIPIndexedBaseTrack(i, info, toptks[i], bottomtks[i]);
                }
                m_Top = new SySal.Scanning.Plate.Side(toptks, rd.TopExt, rd.TopInt);
                m_Bottom = new SySal.Scanning.Plate.Side(bottomtks, rd.BottomInt, rd.BottomExt);
            }
        }

        internal class OPERALinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
        {
            internal class OPERASide : SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side
            {
                public OPERASide(OPERAMIPIndexedEmulsionTrack[] tks, double topz, double bottomz)
                {
                    m_Tracks = tks;
                    m_TopZ = topz;
                    m_BottomZ = bottomz;
                }

                public void SetViews(OPERAView[] vw)
                {
                    m_Views = vw;
                }
            }

            internal class OPERAView : SySal.Scanning.Plate.IO.OPERA.LinkedZone.View
            {
                public void SetSide(OPERASide side) { m_Side = side; }

                public void SetInfo(SySal.BasicTypes.Vector2 mappos, double topz, double bottomz)
                {
                    m_Position = mappos;
                    m_TopZ = topz;
                    m_BottomZ = bottomz;
                }

                public void SetTracks(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[] tks)
                {
                    m_Tracks = tks;
                }

                public OPERAView(int id)
                {
                    m_Id = id;
                }
            }

            internal class OPERAMIPIndexedEmulsionTrack : SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack
            {
                public OPERAMIPIndexedEmulsionTrack(int id, SySal.Tracking.MIPEmulsionTrackInfo info, int frag, int view, int trackid)
                {
                    m_Id = id;
                    m_Info = info;
                    m_OriginalRawData.Fragment = frag;
                    m_OriginalRawData.View = view;
                    m_OriginalRawData.Track = trackid;
                }

                public void SetView(SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw)
                {
                    m_View = vw;
                }
            }

            internal OPERALinkedZone(Catalog Cat, SySal.DAQSystem.Scanning.IntercalibrationInfo TransformInfo, IntMIPIndexedBaseTrack[] Linked, RefDepths rd, OPERAView[][] viewrecords, IntTrack[] TopTracks, IntTrack[] BottomTracks)
                : base()
            {
                int i;
                m_Id = Cat.Id;
                m_Extents = Cat.Extents;
                m_Transform = TransformInfo;
                m_Center.X = (m_Extents.MinX + m_Extents.MaxX) * 0.5f;
                m_Center.Y = (m_Extents.MinY + m_Extents.MaxY) * 0.5f;
                OPERAView topview = null, bottomview = null;
                //System.Collections.ArrayList[] toplist = null;
                ObjList[] toplist = null;
                //System.Collections.ArrayList[] bottomlist = null;
                ObjList[] bottomlist = null;
                if (viewrecords == null)
                {
                    topview = new OPERALinkedZone.OPERAView(0);
                    topview.SetInfo(m_Center, rd.TopExt, rd.TopInt);
                    bottomview = new OPERALinkedZone.OPERAView(0);
                    bottomview.SetInfo(m_Center, rd.BottomInt, rd.BottomExt);
                }
                else
                {
                    //toplist = new System.Collections.ArrayList[viewrecords[0].Length];
                    toplist = new ObjList[viewrecords[0].Length];
/*                    for (i = 0; i < toplist.Length; i++)
                        toplist[i] = new System.Collections.ArrayList();*/
                    //bottomlist = new System.Collections.ArrayList[viewrecords[1].Length];
                    bottomlist = new ObjList[viewrecords[1].Length];
/*                    for (i = 0; i < bottomlist.Length; i++)
                        bottomlist[i] = new System.Collections.ArrayList();*/
                }
                OPERAMIPIndexedEmulsionTrack[] toptks = null;
                if (TopTracks == null)
                {
                    toptks = new OPERAMIPIndexedEmulsionTrack[Linked.Length];
                    for (i = 0; i < toptks.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo tinfo = new MIPEmulsionTrackInfo();
                        IntTrack t = Linked[i].IntTop;
                        tinfo.AreaSum = t.AreaSum;
                        tinfo.Count = (ushort)t.Count;
                        tinfo.Intercept = t.Intercept;
                        tinfo.Slope = t.Slope;
                        tinfo.Sigma = t.Sigma;
                        tinfo.TopZ = t.TopZ;
                        tinfo.BottomZ = t.BottomZ;
                        double dz = (double)rd.TopInt - t.Intercept.Z;
                        tinfo.Field = 0;
                        tinfo.Intercept.Z += dz;
                        tinfo.TopZ += dz;
                        tinfo.BottomZ += dz;
                        t.NewTrack = toptks[i] = new OPERAMIPIndexedEmulsionTrack(i, tinfo,
                            (int)t.OriginalFragment, t.OriginalView, t.OriginalIndex);
                        if (viewrecords != null)
                        {
                            //toplist[t.ViewId].Add(((OPERAMIPIndexedEmulsionTrack)t.NewTrack));
                            toplist[t.ViewId] = new ObjList(t.NewTrack, toplist[t.ViewId]);
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(viewrecords[0][t.ViewId]);                            
                        }
                        else
                        {
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(topview);
                        }
                    }
                }
                else
                {
                    toptks = new OPERAMIPIndexedEmulsionTrack[TopTracks.Length];
                    for (i = 0; i < toptks.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo tinfo = new MIPEmulsionTrackInfo();
                        IntTrack t = TopTracks[i];
                        tinfo.AreaSum = t.AreaSum;
                        tinfo.Count = (ushort)t.Count;
                        tinfo.Intercept = t.Intercept;
                        tinfo.Slope = t.Slope;
                        tinfo.Sigma = t.Sigma;
                        tinfo.TopZ = t.TopZ;
                        tinfo.BottomZ = t.BottomZ;
                        double dz = (double)rd.TopInt - t.Intercept.Z;
                        tinfo.Field = 0;
                        tinfo.Intercept.Z += dz;
                        tinfo.TopZ += dz;
                        tinfo.BottomZ += dz;
                        t.NewTrack = toptks[i] = new OPERAMIPIndexedEmulsionTrack(i, tinfo,
                            (int)t.OriginalFragment, t.OriginalView, t.OriginalIndex);                        
                        if (viewrecords != null)
                        {
                            //toplist[t.ViewId].Add(((OPERAMIPIndexedEmulsionTrack)t.NewTrack));
                            toplist[t.ViewId] = new ObjList(t.NewTrack, toplist[t.ViewId]);
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(viewrecords[0][t.ViewId]);                            
                        }
                        else
                        {
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(topview);
                        }
                        TopTracks[i] = null;
                    }
                }

                OPERAMIPIndexedEmulsionTrack[] bottomtks = null;
                if (BottomTracks == null)
                {
                    bottomtks = new OPERAMIPIndexedEmulsionTrack[Linked.Length];
                    for (i = 0; i < bottomtks.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo binfo = new MIPEmulsionTrackInfo();
                        IntTrack t = Linked[i].IntBottom;
                        binfo.AreaSum = t.AreaSum;
                        binfo.Count = (ushort)t.Count;
                        binfo.Intercept = t.Intercept;
                        binfo.Slope = t.Slope;
                        binfo.Sigma = t.Sigma;
                        binfo.TopZ = t.TopZ;
                        binfo.BottomZ = t.BottomZ;
                        double bdz = (double)rd.BottomInt - t.Intercept.Z;
                        binfo.Field = 0;
                        binfo.Intercept.Z += bdz;
                        binfo.TopZ += bdz;
                        binfo.BottomZ += bdz;
                        t.NewTrack = bottomtks[i] = new OPERAMIPIndexedEmulsionTrack(i, binfo,
                            (int)t.OriginalFragment, t.OriginalView, t.OriginalIndex);
                        if (viewrecords != null)
                        {
                            //bottomlist[t.ViewId].Add(((OPERAMIPIndexedEmulsionTrack)t.NewTrack));
                            bottomlist[t.ViewId] = new ObjList(t.NewTrack, toplist[t.ViewId]);
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(viewrecords[1][t.ViewId]);                            
                        }
                        else
                        {
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(bottomview);
                        }
                    }
                }
                else
                {
                    bottomtks = new OPERAMIPIndexedEmulsionTrack[BottomTracks.Length];
                    for (i = 0; i < bottomtks.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo binfo = new MIPEmulsionTrackInfo();
                        IntTrack t = BottomTracks[i];
                        binfo.AreaSum = t.AreaSum;
                        binfo.Count = (ushort)t.Count;
                        binfo.Intercept = t.Intercept;
                        binfo.Slope = t.Slope;
                        binfo.Sigma = t.Sigma;
                        binfo.TopZ = t.TopZ;
                        binfo.BottomZ = t.BottomZ;
                        double bdz = (double)rd.BottomInt - t.Intercept.Z;
                        binfo.Field = 0;
                        binfo.Intercept.Z += bdz;
                        binfo.TopZ += bdz;
                        binfo.BottomZ += bdz;
                        t.NewTrack = bottomtks[i] = new OPERAMIPIndexedEmulsionTrack(i, binfo,
                            (int)t.OriginalFragment, t.OriginalView, t.OriginalIndex);
                        if (viewrecords != null)
                        {
                            //bottomlist[t.ViewId].Add(((OPERAMIPIndexedEmulsionTrack)t.NewTrack));
                            bottomlist[t.ViewId] = new ObjList(t.NewTrack, toplist[t.ViewId]);
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(viewrecords[1][t.ViewId]);                            
                        }
                        else
                        {
                            ((OPERAMIPIndexedEmulsionTrack)t.NewTrack).SetView(bottomview);
                        }
                        BottomTracks[i] = null;
                    }
                }

                SySal.Scanning.MIPBaseTrack[] basetks = new SySal.Scanning.MIPBaseTrack[Linked.Length];
                for (i = 0; i < Linked.Length; i++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = Linked[i].Info;
                    info.Field = 0;
                    double dz = (double)rd.TopInt - info.Intercept.Z;
                    info.Intercept.Z += dz;
                    info.TopZ += dz;
                    info.BottomZ += dz;
                    basetks[i] = new IntMIPIndexedBaseTrack(i, info, Linked[i].IntTop.NewTrack, Linked[i].IntBottom.NewTrack);
                    Linked[i] = null;
                }
                if (viewrecords == null)
                {
                    topview.SetTracks(toptks);
                    bottomview.SetTracks(bottomtks);
                }
                else
                {
                    for (i = 0; i < toplist.Length; i++)
                    {
                        viewrecords[0][i].SetTracks((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[])ObjList.ToArray(toplist[i], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)));
                        toplist[i] = null;
                    }
                    for (i = 0; i < bottomlist.Length; i++)
                    {
                        viewrecords[1][i].SetTracks((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[])ObjList.ToArray(bottomlist[i], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)));
                        bottomlist[i] = null;
                    }
                    /*
                    for (i = 0; i < toplist.Length; i++)
                    {
                        viewrecords[0][i].SetTracks((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[])toplist[i].ToArray(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)));
                        toplist[i] = null;
                    }
                    for (i = 0; i < bottomlist.Length; i++)
                    {
                        viewrecords[1][i].SetTracks((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[])bottomlist[i].ToArray(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)));
                        bottomlist[i] = null;
                    }
                     */
                }
                m_Top = new OPERALinkedZone.OPERASide(toptks, rd.TopExt, rd.TopInt);
                m_Bottom = new OPERALinkedZone.OPERASide(bottomtks, rd.BottomInt, rd.BottomExt);
                if (viewrecords == null)
                {
                    topview.SetSide((OPERALinkedZone.OPERASide)m_Top);
                    bottomview.SetSide((OPERALinkedZone.OPERASide)m_Bottom);
                    ((OPERALinkedZone.OPERASide)m_Top).SetViews(new OPERALinkedZone.OPERAView[1] { topview });
                    ((OPERALinkedZone.OPERASide)m_Bottom).SetViews(new OPERALinkedZone.OPERAView[1] { bottomview });
                }
                else
                {
                    for (i = 0; i < toplist.Length; i++)
                        viewrecords[0][i].SetSide((OPERALinkedZone.OPERASide)m_Top);
                    ((OPERALinkedZone.OPERASide)m_Top).SetViews(viewrecords[0]);
                    for (i = 0; i < bottomlist.Length; i++)
                        viewrecords[1][i].SetSide((OPERALinkedZone.OPERASide)m_Bottom);
                    ((OPERALinkedZone.OPERASide)m_Bottom).SetViews(viewrecords[1]);
                }
                m_Tracks = basetks;
            }
        }

        private class CHORUSLinkedZone : SySal.Scanning.Plate.IO.CHORUS.LinkedZone
        {
            internal CHORUSLinkedZone(Catalog Cat, IntMIPIndexedBaseTrack[] Linked, RefDepths rd)
                : base()
            {
                int i;
                m_Id = Cat.Id;
                m_Center.X = (Cat.Extents.MinX + Cat.Extents.MaxX) * 0.5f;
                m_Center.Y = (Cat.Extents.MinY + Cat.Extents.MaxY) * 0.5f;
                m_PredictedSlope.X = m_PredictedSlope.Y = 0.0f;
                m_SideSlopeTolerance = m_GlobalSlopeTolerance = m_GoodSlopeTolerance = 0;
                m_Fields = new SySal.Scanning.Plate.IO.CHORUS.FieldHistory[1];
                m_Fields[0].Top = SySal.Scanning.Plate.IO.CHORUS.FieldFlag.OK;
                m_Fields[0].Bottom = SySal.Scanning.Plate.IO.CHORUS.FieldFlag.OK;
                SySal.Scanning.MIPIndexedEmulsionTrack[] toptks = new SySal.Scanning.MIPIndexedEmulsionTrack[Linked.Length];
                SySal.Scanning.MIPIndexedEmulsionTrack[] bottomtks = new SySal.Scanning.MIPIndexedEmulsionTrack[Linked.Length];
                SySal.Scanning.MIPBaseTrack[] basetks = new SySal.Scanning.MIPBaseTrack[Linked.Length];
                for (i = 0; i < Linked.Length; i++)
                {

                    /*
                                                                    tinfo.AreaSum = t.AreaSum;
                                                                    tinfo.Count = (ushort)t.Count;
                                                                    tinfo.Intercept = t.Intercept;
                                                                    tinfo.Slope = t.Slope;
                                                                    tinfo.Sigma = t.Sigma;
                                                                    tinfo.TopZ = t.TopZ;
                                                                    tinfo.BottomZ = t.BottomZ;
												
                                                                    binfo.AreaSum = tt.AreaSum;
                                                                    binfo.Count = (ushort)tt.Count;
                                                                    binfo.Intercept = tt.Intercept;
                                                                    binfo.Slope = tt.Slope;
                                                                    binfo.Sigma = tt.Sigma;
                                                                    binfo.TopZ = tt.TopZ;
                                                                    binfo.BottomZ = tt.BottomZ;

  
                     */
                    SySal.Tracking.MIPEmulsionTrackInfo tinfo = new MIPEmulsionTrackInfo();
                    IntTrack t = Linked[i].IntTop;
                    tinfo.AreaSum = t.AreaSum;
                    tinfo.Count = (ushort)t.Count;
                    tinfo.Intercept = t.Intercept;
                    tinfo.Slope = t.Slope;
                    tinfo.Sigma = t.Sigma;
                    tinfo.TopZ = t.TopZ;
                    tinfo.BottomZ = t.BottomZ;
                    double dz = (double)rd.TopInt - tinfo.Intercept.Z;
                    tinfo.Field = 0;
                    tinfo.Intercept.Z += dz;
                    tinfo.TopZ += dz;
                    tinfo.BottomZ += dz;
                    toptks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(tinfo, null, i);

                    SySal.Tracking.MIPEmulsionTrackInfo binfo = new MIPEmulsionTrackInfo();
                    t = Linked[i].IntBottom;
                    binfo.AreaSum = t.AreaSum;
                    binfo.Count = (ushort)t.Count;
                    binfo.Intercept = t.Intercept;
                    binfo.Slope = t.Slope;
                    binfo.Sigma = t.Sigma;
                    binfo.TopZ = t.TopZ;
                    binfo.BottomZ = t.BottomZ;
                    binfo.Field = 0;
                    binfo.Intercept.Z += dz;
                    binfo.TopZ += dz;
                    binfo.BottomZ += dz;
                    bottomtks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(binfo, null, i);
                    t = null;

#if false	//cut here
					SySal.Tracking.MIPEmulsionTrackInfo tinfo = Linked[i].Top.Info;
					double dz = (double)rd.TopInt - tinfo.Intercept.Z;
					tinfo.Field = 0;
					tinfo.Intercept.Z += dz;
					tinfo.TopZ += dz;
					tinfo.BottomZ += dz;
					toptks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(tinfo, null, i);
					SySal.Tracking.MIPEmulsionTrackInfo binfo = Linked[i].Bottom.Info;
					binfo.Field = 0;
					binfo.Intercept.Z += dz;
					binfo.TopZ += dz;
					binfo.BottomZ += dz;					
					bottomtks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(binfo, null, i);
#endif
                    SySal.Tracking.MIPEmulsionTrackInfo info = Linked[i].Info;
                    info.Field = 0;
                    info.Intercept.Z += dz;
                    info.TopZ += dz;
                    info.BottomZ += dz;
                    basetks[i] = new IntMIPIndexedBaseTrack(i, info, toptks[i], bottomtks[i]);
                    // new
                    Linked[i] = null;
                    // end new
                }
                m_Top = new SySal.Scanning.Plate.Side(toptks, rd.TopExt, rd.TopInt);
                m_Bottom = new SySal.Scanning.Plate.Side(bottomtks, rd.BottomInt, rd.BottomExt);
                m_Tracks = basetks;
            }
        }
        #endregion

        public dShouldStop ShouldStop
        {
            get
            {
                return intShouldStop;
            }
            set
            {
                intShouldStop = value;
            }
        }

        public dLoadFragment Load
        {
            get
            {
                return intLoad;
            }
            set
            {
                intLoad = value;
            }
        }

        public dProgress Progress
        {
            get
            {
                return intProgress;
            }
            set
            {
                intProgress = value;
            }
        }

        internal class IntMIPIndexedBaseTrack : SySal.Scanning.MIPBaseTrack
        {
            public IntTrack IntTop, IntBottom;
            public IntMIPIndexedBaseTrack() { }
            public IntMIPIndexedBaseTrack(int id, SySal.Tracking.MIPEmulsionTrackInfo info, SySal.Scanning.MIPIndexedEmulsionTrack toptk, SySal.Scanning.MIPIndexedEmulsionTrack bottomtk)
            {
                m_Id = id;
                m_Info = info;
                m_Top = toptk;
                m_Bottom = bottomtk;
            }
        }

        private class ObjList
        {
            public object Info;
            public ObjList Next;
            public ObjList(object o, ObjList n) { Info = o; Next = n; }
            public static object ToArray(ObjList u, System.Type t)
            {
                int len = 0;
                ObjList n = u;
                while (n != null)
                {
                    len++;
                    n = n.Next;
                }
                Array a = Array.CreateInstance(t, len);
                len = 0;
                n = u;
                while (n != null)
                {
                    a.SetValue(n.Info, len++);
                    n = n.Next;
                }
                return a;
            }
        }

        const string LogHeaderString = "TVID TVIEW TOF TOV TOI TN TA TPX TPY TPZ TSX TSY TS TTZ TBZ TZB TZE TOFPX TOFPY TOFPZ TOSX TOSY TOSZ BVID BVIEW BOF BOV BOI BN BA BPX BPY BPZ BSX BSY TB BTZ BBZ BZB BZE BOFPX BOFPY BOFPZ BOSX BOSY BOSZ N A PX PY PZ SX SY S";

        /// <summary>
        /// Links RWD files into a LinkedZone, producing an object of the specified output type.
        /// </summary>
        /// <param name="Cat">the Catalog of the Raw Data Files.</param>
        /// <param name="outputtype">the type of output to be produced. Currently, the following formats are supported:
        /// <list type="table">
        /// <item><term>SySal.Scanning.Plate.IO.CHORUS.LinkedZone</term><description>CHORUS - style Linked Zone. This format is quite obsolete.</description></item>
        /// <item><term>SySal.Scanning.Plate.IO.OPERA.LinkedZone</term><description>OPERA - style Linked Zone. This is the format that supports all current options.</description></item>
        /// </list>
        /// </param>
        /// <returns>the LinkedZone produced.</returns>
        public SySal.Scanning.Plate.LinkedZone Link(Catalog Cat, System.Type outputtype)
        {
            switch (outputtype.FullName)
            {
                case "SySal.Scanning.Plate.LinkedZone": break;
                case "SySal.Scanning.Plate.IO.CHORUS.LinkedZone": break;
                case "SySal.Scanning.Plate.IO.OPERA.LinkedZone": break;
                case "SySal.DataStreams.OPERALinkedZone": break;
                default: throw new SySal.Scanning.PostProcessing.FragmentLinking.LinkException("Unsupported plate type");
            }
        
            OPERALinkedZone.OPERAView[][] viewrecords = null;
            if (C.PreserveViews)
            {
                viewrecords = new OPERALinkedZone.OPERAView[2][];
                viewrecords[0] = new OPERALinkedZone.OPERAView[Cat.XSize * Cat.YSize];
                viewrecords[1] = new OPERALinkedZone.OPERAView[Cat.XSize * Cat.YSize];
            }

            Console.WriteLine("Dump file path: \"" + C.DumpFilePath + "\".");
            System.IO.StreamWriter logger = null;
            if (C.DumpFilePath.Length > 0)
            {
                logger = new System.IO.StreamWriter(C.DumpFilePath);
                logger.WriteLine(LogHeaderString);
            }

            try
            {
                RefDepths rd = new RefDepths();
                rd.TopExt = 0.0f;
                rd.TopInt = 0.0f;
                rd.BottomInt = 0.0f;
                rd.BottomExt = 0.0f;
                rd.Hits = 0;
                IntMIPIndexedBaseTrack[] Linked = null;
                IntTrack[] TopTracks = null;
                IntTrack[] BottomTracks = null;
                /*ArrayList templinked = new ArrayList(10000);*/
                /*ArrayList temptop = new ArrayList(10000);*/
                /*ArrayList tempbottom = new ArrayList(10000);*/
                ObjList templinked = null;
                ObjList temptop = null;
                ObjList tempbottom = null;
                bool TransformSet = false;
                SySal.DAQSystem.Scanning.IntercalibrationInfo TransformInfo = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                TransformInfo.MXX = TransformInfo.MYY = 1.0;
                TransformInfo.MXY = TransformInfo.MYX = 0.0;
                TransformInfo.RX = TransformInfo.RY = 0.0;
                TransformInfo.TX = TransformInfo.TX = 0.0;

                int YSize, XSize;
                YSize = (int)Cat.YSize;
                XSize = (int)Cat.XSize;

                IntTileLine[] TileLines = new IntTileLine[YSize];
                bool[] FragmentLoaded = new bool[Cat.Fragments];
                int i, j, k, ix, iy;
                int basecount = 0;
                int processorcount = C.SingleThread ? 1 : Environment.ProcessorCount;
                dQualityCut[] QC = new dQualityCut[processorcount];
                for (i = 0; i < QC.Length; i++)
                    QC[i] = new dQualityCut(new QCFunctor(C.QualityCut).QCApply);

                for (iy = 0; iy < YSize; iy++)
                {
                    if (intProgress != null) intProgress((double)iy / (double)YSize * 50.0f);
                    if (intShouldStop != null)
                        if (intShouldStop()) return null;

                    for (ix = 0; ix < XSize; ix++)
                        LoadFragment(Cat[iy, ix], FragmentLoaded, TileLines, XSize, viewrecords, ref TransformSet, ref TransformInfo);

                    IntTileLine t = TileLines[iy];
                    for (i = 0; i < t.Fill; i++)
                    {
                        int iix, iiy, minvx, maxvx, kk;
                        IntFragment pf = t.Fragments[i];
                        minvx = -1;
                        for (j = 0; j < pf.Views.Length; j++)
                        {
                            maxvx = XSize + 1;
                            kk = 0;
                            for (k = 0; k < pf.Views.Length; k++)
                                if (pf.Views[k].TileX < maxvx && pf.Views[k].TileX > minvx)
                                {
                                    kk = k;
                                    maxvx = pf.Views[kk].TileX;
                                }
                            minvx = pf.Views[kk].TileX;
                            IntView v = pf.Views[kk];
                            if (v.IsOnDisk) v.ReloadFromDisk();
                            rd.TopExt += v.Top.ZExt;
                            rd.TopInt += v.Top.ZBase;
                            rd.BottomExt += v.Bottom.ZExt;
                            rd.BottomInt += v.Bottom.ZBase;
                            rd.Hits++;
                            for (iiy = iy; iiy <= iy + 1; iiy++)
                                if (iiy < YSize)
                                    for (iix = (iiy > iy && v.TileX > 0) ? (v.TileX - 1) : v.TileX; iix <= v.TileX + 1; iix++)
                                        if (iix < XSize)
                                        {
                                            uint indexneeded = Cat[iiy, iix];
                                            LoadFragment(indexneeded, FragmentLoaded, TileLines, XSize, viewrecords, ref TransformSet, ref TransformInfo);
                                            for (k = 0; (k < TileLines[iiy].Fill) && (indexneeded != TileLines[iiy].Fragments[k].Index); k++) ;
                                            IntFragment pff = TileLines[iiy].Fragments[k];
                                            for (k = 0; (k < pff.Views.Length) && (pff.Views[k].TileX != iix); k++) ;
                                            IntView w = pff.Views[k];
                                            v.Clean(w, C.PosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.SingleThread);
                                            if (C.MemorySaving >= 3 && v != w) w.MoveToDisk();
                                        }
                        }
                    }
                }
                for (iy = 0; iy < YSize; iy++)
                {
                    if (intProgress != null) intProgress(50.0f + (double)iy / (double)YSize * 50.0f);
                    if (intShouldStop != null)
                        if (intShouldStop()) return null;

                    IntTileLine t = TileLines[iy];
                    for (i = 0; i < t.Fill; i++)
                    {
                        int iix, iiy, minvx, maxvx, kk;
                        IntFragment pf = t.Fragments[i];
                        minvx = -1;
                        for (j = 0; j < pf.Views.Length; j++)
                        {
                            maxvx = XSize + 1;
                            kk = 0;
                            for (k = 0; k < pf.Views.Length; k++)
                                if (pf.Views[k].TileX < maxvx && pf.Views[k].TileX > minvx)
                                {
                                    kk = k;
                                    maxvx = pf.Views[kk].TileX;
                                }
                            minvx = pf.Views[kk].TileX;
                            IntView v = pf.Views[kk];
                            if (C.KeepLinkedTracksOnly == false)
                            {
                                foreach (IntTrack tk in v.Top.Tracks)
                                    if (tk.Valid)//    temptop.Add(tk);
                                        temptop = new ObjList(tk, temptop);
                                foreach (IntTrack tk in v.Bottom.Tracks)
                                    if (tk.Valid)//    tempbottom.Add(tk);
                                        tempbottom = new ObjList(tk, tempbottom);
                            }
                            for (iiy = iy; iiy <= iy + 1; iiy++)
                                if (iiy < YSize)
                                    for (iix = (iiy > iy && v.TileX > 0) ? (v.TileX - 1) : v.TileX; iix <= v.TileX + 1; iix++)
                                        if (iix < XSize)
                                        {
                                            uint indexneeded = Cat[iiy, iix];
                                            LoadFragment(indexneeded, FragmentLoaded, TileLines, XSize, viewrecords, ref TransformSet, ref TransformInfo);
                                            for (k = 0; (k < TileLines[iiy].Fill) && (indexneeded != TileLines[iiy].Fragments[k].Index); k++) ;
                                            IntFragment pff = TileLines[iiy].Fragments[k];
                                            for (k = 0; (k < pff.Views.Length) && (pff.Views[k].TileX != iix); k++) ;
                                            IntView w = pff.Views[k];
                                            System.Collections.ArrayList ahr = v.Link(w, C.MinGrains, C.MinSlope * C.MinSlope, C.PosTol, C.SlopeTol, C.SlopeTolIncreaseWithSlope, QC, C.PreventDuplication, C.LinkLimit, null, false, ref basecount, logger);
                                            foreach (object o in ahr)
                                                templinked = new ObjList(o, templinked);
                                            //templinked.AddRange(v.Link(w, C.MinGrains, C.MinSlope * C.MinSlope, C.PosTol, C.SlopeTol, C.SlopeTolIncreaseWithSlope, QC, C.PreventDuplication));
                                            if (C.KeepLinkedTracksOnly && C.MemorySaving >= 3 && v != w) w.MoveToDisk();
                                        }
                            if (C.KeepLinkedTracksOnly && C.MemorySaving >= 2) v.MoveToDisk();
                        }
                    }
                    for (i = 0; i < t.Fill; i++)
                    {
                        IntFragment ifr = t.Fragments[i];
                        foreach (IntView ivw in ifr.Views)
                            if (ivw.IsOnDisk == false) ivw.MoveToDisk();
                    }
                    IntTileLine itl = TileLines[iy];
                    if (itl != null)
                    {
                        if (itl.Fragments != null)
                            foreach (IntFragment ifr in itl.Fragments)
                                if (ifr != null && ifr.Views != null)
                                    foreach (IntView ivw in ifr.Views)
                                        if (ivw != null)
                                        {
                                            if (PF != null && C.KeepLinkedTracksOnly == true) // templinked.AddRange(ivw.PromoteMicrotracks(PC));
                                            {
                                                int dummy = 0;
                                                System.Collections.ArrayList ahr = ivw.PromoteMicrotracks(PC, null, C.LinkLimit, ref dummy);
                                                foreach (object o in ahr)
                                                    templinked = new ObjList(o, templinked);
                                            }
                                            ivw.Dispose();
                                        }
                        TileLines[iy] = null;
                    }
                }
                TileLines = null;
                //Linked = (IntMIPIndexedBaseTrack[])templinked.ToArray(typeof(IntMIPIndexedBaseTrack));
                Linked = (IntMIPIndexedBaseTrack[])ObjList.ToArray(templinked, typeof(IntMIPIndexedBaseTrack));
                templinked = null;
                if (C.KeepLinkedTracksOnly == false)
                {
                    //TopTracks = (IntTrack[])temptop.ToArray(typeof(IntTrack));
                    TopTracks = (IntTrack[])ObjList.ToArray(temptop, typeof(IntTrack));
                    //BottomTracks = (IntTrack[])tempbottom.ToArray(typeof(IntTrack));
                    BottomTracks = (IntTrack[])ObjList.ToArray(tempbottom, typeof(IntTrack));
                }
                GC.Collect();
                if (intProgress != null) intProgress(100.0f);
                rd.TopExt /= rd.Hits;
                rd.TopInt /= rd.Hits;
                rd.BottomInt /= rd.Hits;
                rd.BottomExt /= rd.Hits;

                switch (outputtype.FullName)
                {
                    case "SySal.Scanning.Plate.IO.CHORUS.LinkedZone": return new StripesFragLink2.StripesFragmentLinker.CHORUSLinkedZone(Cat, Linked, rd);
                    case "SySal.Scanning.Plate.IO.OPERA.LinkedZone": return new StripesFragLink2.StripesFragmentLinker.OPERALinkedZone(Cat, TransformInfo, Linked, rd, viewrecords, TopTracks, BottomTracks);
                }
                return new StripesFragLink2.StripesFragmentLinker.LinkedZone(Cat, Linked, rd);
            }
            finally
            {
                if (logger != null)
                {
                    logger.Flush();
                    logger.Close();
                }
            }
        }

        class MyView : SySal.Scanning.Plate.IO.OPERA.LinkedZone.View
        {
            public MyView(IntView v, bool istop)
            {
                if (istop)
                {
                    this.m_Id = v.Top.Id;
                    this.m_Position = v.Top.MapPos;
                    this.m_TopZ = v.Top.ZExt;
                    this.m_BottomZ = v.Top.ZBase;                   
                }
                else
                {
                    this.m_Id = v.Bottom.Id;
                    this.m_Position = v.Bottom.MapPos;
                    this.m_TopZ = v.Bottom.ZBase; 
                    this.m_BottomZ = v.Bottom.ZExt; 
                }
            }
        }

        /// <summary>
        /// Links RWD files into a LinkedZone, producing a file of the specified output type.
        /// </summary>
        /// <param name="Cat">the Catalog of the Raw Data Files.</param>
        /// <param name="outputtype">the type of output to be produced. Currently, the following formats are supported:
        /// <list type="table">
        /// <item><term>SySal.DataStreams.OPERALinkedZone</term><description>OPERA - style Linked Zone (microtracks only).</description></item>
        /// </list>
        /// </param>
        /// <param name="outfilepath">the path of the output file.</param>
        public void LinkToFile(Catalog Cat, System.Type outputtype, string outfilepath)
        {
            switch (outputtype.FullName)
            {
                case "SySal.DataStreams.OPERALinkedZone": break;
                default: throw new SySal.Scanning.PostProcessing.FragmentLinking.LinkException("Unsupported plate type");
            }

            OPERALinkedZone.OPERAView[][] viewrecords = null;
            if (C.PreserveViews)
            {
                viewrecords = new OPERALinkedZone.OPERAView[2][];
                viewrecords[0] = new OPERALinkedZone.OPERAView[Cat.XSize * Cat.YSize];
                viewrecords[1] = new OPERALinkedZone.OPERAView[Cat.XSize * Cat.YSize];
            }

            Console.WriteLine("Dump file path: \"" + C.DumpFilePath + "\".");
            System.IO.StreamWriter logger = null;
            if (C.DumpFilePath.Length > 0)
            {
                logger = new System.IO.StreamWriter(C.DumpFilePath);
                logger.WriteLine(LogHeaderString);
            }

            try
            {
                RefDepths rd = new RefDepths();
                rd.TopExt = 0.0f;
                rd.TopInt = 0.0f;
                rd.BottomInt = 0.0f;
                rd.BottomExt = 0.0f;
                rd.Hits = 0;
                bool TransformSet = false;
                SySal.DAQSystem.Scanning.IntercalibrationInfo txinfo = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                txinfo.MXX = txinfo.MYY = 1.0;
                txinfo.MXY = txinfo.MYX = 0.0;
                txinfo.RX = txinfo.RY = 0.0;
                txinfo.TX = txinfo.TX = 0.0;

                int YSize, XSize;
                YSize = (int)Cat.YSize;
                XSize = (int)Cat.XSize;

                IntTileLine[] TileLines = new IntTileLine[YSize];
                bool[] FragmentLoaded = new bool[Cat.Fragments];
                int i, j, k, ix, iy;
                int processorcount = C.SingleThread ? 1 : Environment.ProcessorCount;
                dQualityCut[] QC = new dQualityCut[processorcount];
                for (i = 0; i < QC.Length; i++)
                    QC[i] = new dQualityCut(new QCFunctor(C.QualityCut).QCApply);

                for (iy = 0; iy < YSize; iy++)
                {
                    if (intProgress != null) intProgress((double)iy / (double)YSize * 50.0f);
                    if (intShouldStop != null)
                        if (intShouldStop()) return;

                    for (ix = 0; ix < XSize; ix++)
                        LoadFragment(Cat[iy, ix], FragmentLoaded, TileLines, XSize, viewrecords, ref TransformSet, ref txinfo);

                    IntTileLine t = TileLines[iy];
                    for (i = 0; i < t.Fill; i++)
                    {
                        int iix, iiy, minvx, maxvx, kk;
                        IntFragment pf = t.Fragments[i];
                        minvx = -1;
                        for (j = 0; j < pf.Views.Length; j++)
                        {
                            maxvx = XSize + 1;
                            kk = 0;
                            for (k = 0; k < pf.Views.Length; k++)
                                if (pf.Views[k].TileX < maxvx && pf.Views[k].TileX > minvx)
                                {
                                    kk = k;
                                    maxvx = pf.Views[kk].TileX;
                                }
                            minvx = pf.Views[kk].TileX;
                            IntView v = pf.Views[kk];
                            rd.TopExt += v.Top.ZExt;
                            rd.TopInt += v.Top.ZBase;
                            rd.BottomExt += v.Bottom.ZExt;
                            rd.BottomInt += v.Bottom.ZBase;
                            rd.Hits++;
                            for (iiy = iy; iiy <= iy + 1; iiy++)
                                if (iiy < YSize)
                                    for (iix = (iiy > iy && v.TileX > 0) ? (v.TileX - 1) : v.TileX; iix <= v.TileX + 1; iix++)
                                        if (iix < XSize)
                                        {
                                            uint indexneeded = Cat[iiy, iix];
                                            LoadFragment(indexneeded, FragmentLoaded, TileLines, XSize, viewrecords, ref TransformSet, ref txinfo);
                                            for (k = 0; (k < TileLines[iiy].Fill) && (indexneeded != TileLines[iiy].Fragments[k].Index); k++) ;
                                            IntFragment pff = TileLines[iiy].Fragments[k];
                                            for (k = 0; (k < pff.Views.Length) && (pff.Views[k].TileX != iix); k++) ;
                                            IntView w = pff.Views[k];
                                            v.Clean(w, C.PosTol, C.MergePosTol * C.MergePosTol, C.MergeSlopeTol * C.MergeSlopeTol, C.SingleThread);
                                            if (C.MemorySaving >= 3 && iiy > iy) w.MoveToDisk();
                                        }
                            if (C.KeepLinkedTracksOnly && C.MemorySaving >= 2) v.MoveToDisk();
                        }
                    }
                    for (i = 0; i < t.Fill; i++)
                    {
                        IntFragment ifr = t.Fragments[i];
                        foreach (IntView ivw in ifr.Views)
                            if (ivw.IsOnDisk == false) ivw.MoveToDisk();
                    }
                }
                if (intProgress != null) intProgress(50.0f);
                int topvalid = 0;
                int bottomvalid = 0;
                int foundondisk = 0;
                int examined = 0;
                foreach (IntTileLine itl in TileLines)
                    for (i = 0; i < itl.Fill; i++)
                    {
                        IntFragment ifr = itl.Fragments[i];
                        foreach (IntView ivw in ifr.Views)
                        {
                            examined++;
                            if (ivw.IsOnDisk)
                            {
                                foundondisk++;
                                ivw.ReloadFromDisk();
                            }
                            IntSide isi;
                            int itki;
                            isi = ivw.Top;
                            for (itki = 0; itki < isi.Tracks.Length; itki++)
                                if (isi.Tracks[itki].Valid)
                                    isi.Tracks[itki].ValidId = topvalid++;
                            isi = ivw.Bottom;
                            for (itki = 0; itki < isi.Tracks.Length; itki++)
                                if (isi.Tracks[itki].Valid)
                                    isi.Tracks[itki].ValidId = bottomvalid++;
                            ivw.MoveToDisk();
                        }
                    }

                rd.TopExt /= rd.Hits;
                rd.TopInt /= rd.Hits;
                rd.BottomInt /= rd.Hits;
                rd.BottomExt /= rd.Hits;

                SySal.BasicTypes.Vector2 center = new SySal.BasicTypes.Vector2();
                center.X = 0.5 * (Cat.Extents.MinX + Cat.Extents.MaxX);
                center.Y = 0.5 * (Cat.Extents.MinY + Cat.Extents.MaxY);
                SySal.DataStreams.OPERALinkedZone.Writer wr = new SySal.DataStreams.OPERALinkedZone.Writer(outfilepath, Cat.Id, Cat.Extents, center, txinfo);
                wr.SetZInfo(rd.TopExt, rd.TopInt, rd.BottomInt, rd.BottomExt);


                SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry tie = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry();

                int basecount = 0;

                for (iy = 0; iy < YSize; iy++)
                {
                    if (intProgress != null) intProgress(50.0 + (double)iy / (double)YSize * 50.0f);
                    if (intShouldStop != null)
                        if (intShouldStop()) return;

                    IntTileLine t = TileLines[iy];
                    for (i = 0; i < t.Fill; i++)
                    {
                        int iix, iiy, minvx, maxvx, kk;
                        IntFragment pf = t.Fragments[i];
                        minvx = -1;
                        for (j = 0; j < pf.Views.Length; j++)
                        {
                            maxvx = XSize + 1;
                            kk = 0;
                            for (k = 0; k < pf.Views.Length; k++)
                                if (pf.Views[k].TileX < maxvx && pf.Views[k].TileX > minvx)
                                {
                                    kk = k;
                                    maxvx = pf.Views[kk].TileX;
                                }
                            minvx = pf.Views[kk].TileX;
                            IntView v = pf.Views[kk];
                            if (C.KeepLinkedTracksOnly == false)
                            {
                                IntSide isi = null;
                                v.ReloadFromDisk();
                                wr.AddView(new MyView(v, true), true);
                                isi = v.Top;
                                foreach (IntTrack itk in isi.Tracks)
                                    if (itk.Valid)
                                    {
                                        SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                                        info.Field = 0;
                                        info.Count = (ushort)itk.Count;
                                        info.AreaSum = itk.AreaSum;
                                        info.Intercept = itk.Intercept;
                                        info.Slope = itk.Slope;
                                        info.Sigma = itk.Sigma;
                                        info.TopZ = itk.TopZ;
                                        info.BottomZ = itk.BottomZ;
                                        tie.Fragment = (int)itk.OriginalFragment;
                                        tie.View = itk.OriginalView;
                                        tie.Track = itk.OriginalIndex;
                                        wr.AddMIPEmulsionTrack(info, itk.ValidId, isi.Id, tie, true);
                                    }
                                wr.AddView(new MyView(v, false), false);
                                isi = v.Bottom;
                                foreach (IntTrack itk in isi.Tracks)
                                    if (itk.Valid)
                                    {
                                        SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                                        info.Field = 0;
                                        info.Count = (ushort)itk.Count;
                                        info.AreaSum = itk.AreaSum;
                                        info.Intercept = itk.Intercept;
                                        info.Slope = itk.Slope;
                                        info.Sigma = itk.Sigma;
                                        info.TopZ = itk.TopZ;
                                        info.BottomZ = itk.BottomZ;
                                        tie.Fragment = (int)itk.OriginalFragment;
                                        tie.View = itk.OriginalView;
                                        tie.Track = itk.OriginalIndex;
                                        wr.AddMIPEmulsionTrack(info, itk.ValidId, isi.Id, tie, false);
                                    }
                            }
                            for (iiy = iy; iiy <= iy + 1; iiy++)
                                if (iiy < YSize)
                                    for (iix = (iiy > iy && v.TileX > 0) ? (v.TileX - 1) : v.TileX; iix <= v.TileX + 1; iix++)
                                        if (iix < XSize)
                                        {
                                            uint indexneeded = Cat[iiy, iix];
                                            LoadFragment(indexneeded, FragmentLoaded, TileLines, XSize, viewrecords, ref TransformSet, ref txinfo);
                                            for (k = 0; (k < TileLines[iiy].Fill) && (indexneeded != TileLines[iiy].Fragments[k].Index); k++) ;
                                            IntFragment pff = TileLines[iiy].Fragments[k];
                                            for (k = 0; (k < pff.Views.Length) && (pff.Views[k].TileX != iix); k++) ;
                                            IntView w = pff.Views[k];
                                            v.Link(w, C.MinGrains, C.MinSlope * C.MinSlope, C.PosTol, C.SlopeTol, C.SlopeTolIncreaseWithSlope, QC, C.PreventDuplication, C.LinkLimit, wr, C.KeepLinkedTracksOnly, ref basecount, logger);
                                            if (C.KeepLinkedTracksOnly && C.MemorySaving >= 3 && iiy > iy) w.MoveToDisk();
                                        }
                            if (C.KeepLinkedTracksOnly && C.MemorySaving >= 2) v.MoveToDisk();
                        }
                    }
                    if (PF != null && C.KeepLinkedTracksOnly == true)
                        for (i = 0; i < t.Fill; i++)
                        {
                            IntFragment ifr = t.Fragments[i];
                            foreach (IntView ivw in ifr.Views)
                            {
                                ivw.ReloadFromDisk();
                                ivw.PromoteMicrotracks(PC, wr, C.LinkLimit, ref basecount);
                                ivw.MoveToDisk();
                            }
                        }
                    if (C.KeepLinkedTracksOnly == false)
                        for (i = 0; i < t.Fill; i++)
                        {
                            IntFragment ifr = t.Fragments[i];
                            foreach (IntView ivw in ifr.Views)
                                if (ivw.IsOnDisk == false) ivw.MoveToDisk();
                        }
                }

                int topmutkscount = 0;
                int bottommutkscount = 0;
                for (iy = 0; iy < YSize; iy++)
                {
                    if (intProgress != null) intProgress(50.0 + (double)iy / (double)YSize * 50.0);
                    if (intShouldStop != null)
                        if (intShouldStop()) return;
                    IntTileLine itl = TileLines[iy];
                    if (itl != null)
                    {
                        if (itl.Fragments != null)
                            for (i = 0; i < itl.Fill; i++)
                            {
                                IntFragment ifr = itl.Fragments[i];
                                if (ifr != null && ifr.Views != null)
                                    foreach (IntView ivw in ifr.Views)
                                        if (ivw != null)
                                        {
                                            IntSide isi = null;
                                            //ivw.ReloadFromDisk();
                                            wr.AddView(new MyView(ivw, true), true);
                                            /*                                        if (C.KeepLinkedTracksOnly == false)
                                                                                    {
                                                                                        isi = ivw.Top;
                                                                                        foreach (IntTrack itk in isi.Tracks)
                                                                                            if (itk.Valid)
                                                                                            {
                                                                                                SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                                                                                                info.Field = 0;
                                                                                                info.Count = (ushort)itk.Count;
                                                                                                info.AreaSum = itk.AreaSum;
                                                                                                info.Intercept = itk.Intercept;
                                                                                                info.Slope = itk.Slope;
                                                                                                info.Sigma = itk.Sigma;
                                                                                                info.TopZ = itk.TopZ;
                                                                                                info.BottomZ = itk.BottomZ;
                                                                                                tie.Fragment = (int)itk.OriginalFragment;
                                                                                                tie.View = itk.OriginalView;
                                                                                                tie.Track = itk.OriginalIndex;
                                                                                                wr.AddMIPEmulsionTrack(info, topmutkscount++, isi.Id, tie, true);
                                                                                            }
                                                                                    }*/
                                            wr.AddView(new MyView(ivw, false), false);
                                            /*                                        if (C.KeepLinkedTracksOnly == false)
                                                                                    {
                                                                                        isi = ivw.Bottom;
                                                                                        foreach (IntTrack itk in isi.Tracks)
                                                                                            if (itk.Valid)
                                                                                            {
                                                                                                SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                                                                                                info.Field = 0;
                                                                                                info.Count = (ushort)itk.Count;
                                                                                                info.AreaSum = itk.AreaSum;
                                                                                                info.Intercept = itk.Intercept;
                                                                                                info.Slope = itk.Slope;
                                                                                                info.Sigma = itk.Sigma;
                                                                                                info.TopZ = itk.TopZ;
                                                                                                info.BottomZ = itk.BottomZ;
                                                                                                tie.Fragment = (int)itk.OriginalFragment;
                                                                                                tie.View = itk.OriginalView;
                                                                                                tie.Track = itk.OriginalIndex;
                                                                                                wr.AddMIPEmulsionTrack(info, bottommutkscount++, isi.Id, tie, false);
                                                                                            }                                        
                                                                                    }*/
                                            ivw.Dispose();
                                        }
                            }
                    }
                    TileLines[iy] = null;
                }
                wr.Complete();
            }
            finally
            {
                if (logger != null)
                {
                    logger.Flush();
                    logger.Close();
                }
            }
        }

        #endregion

        #region Internals

        class QCFunctor
        {
            private NumericalTools.Function QF;

            private string[] QFParams = new string[0];

            private dGetParam[] dParams = new dGetParam[0];

            public delegate double dGetParam(IntMIPIndexedBaseTrack tk);

            public double _A(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.AreaSum; }
            public double _TA(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.AreaSum ?? tk.IntTop.AreaSum); }
            public double _BA(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.AreaSum ?? tk.IntBottom.AreaSum); }

            public double _N(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Count; }
            public double _TN(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Count ?? tk.IntTop.Count); }
            public double _BN(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Count ?? tk.IntBottom.Count); }

            public double _S(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Sigma; }
            public double _TS(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Sigma ?? tk.IntTop.Sigma); }
            public double _BS(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Sigma ?? tk.IntBottom.Sigma); }

            public double _PX(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Intercept.X; }
            public double _TPX(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Intercept.X ?? tk.IntTop.Intercept.X); }
            public double _BPX(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Intercept.X ?? tk.IntBottom.Intercept.X); }

            public double _PY(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Intercept.Y; }
            public double _TPY(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Intercept.Y ?? tk.IntTop.Intercept.Y); }
            public double _BPY(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Intercept.Y ?? tk.IntBottom.Intercept.Y); }

            public double _PZ(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Intercept.Z; }
            public double _TPZ(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Intercept.Z ?? tk.IntTop.Intercept.Z); }
            public double _BPZ(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Intercept.Z ?? tk.IntBottom.Intercept.Z); }

            public double _SX(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Slope.X; }
            public double _TSX(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Slope.X ?? tk.IntTop.Slope.X); }
            public double _BSX(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Slope.X ?? tk.IntBottom.Slope.X); }

            public double _SY(IntMIPIndexedBaseTrack tk) { return (double)tk.Info.Slope.Y; }
            public double _TSY(IntMIPIndexedBaseTrack tk) { return (double)(tk.Top?.Info.Slope.Y ?? tk.IntTop.Slope.Y); }
            public double _BSY(IntMIPIndexedBaseTrack tk) { return (double)(tk.Bottom?.Info.Slope.Y ?? tk.IntBottom.Slope.Y); }

            public QCFunctor(string fexpr)
            {
                if (string.IsNullOrWhiteSpace(fexpr) == false)
                {
                    QF = new NumericalTools.CStyleParsedFunction(fexpr);
                    QFParams = new string[QF.ParameterList.Length];
                    dParams = new dGetParam[QF.ParameterList.Length];
                    for (int i = 0; i < QFParams.Length; i++)
                    {
                        QFParams[i] = QF.ParameterList[i].ToUpper();
                        try
                        {
                            dParams[i] = (dGetParam)Delegate.CreateDelegate(typeof(dGetParam), this, "_" + QFParams[i]);
                        }
                        catch (Exception x)
                        {
                            throw new Exception("Unknown parameter " + QFParams[i] + System.Environment.NewLine + x.ToString());
                        }
                    }
                }
            }

            public bool QCApply(IntMIPIndexedBaseTrack tk)
            {
                if (QF == null) return true;                
                for (int p = 0; p < dParams.Length; p++)
                    QF[p] = dParams[p](tk);
                return (QF.Evaluate() != 0.0);
            }
        }

        private bool PCApply(IntTrack tk)
        {
            if (PF == null) return false;
            int i, p;
            for (p = 0; p < PFParams.Length; p++)
            {
                switch (PFParams[p])
                {
                    case "A": PF[p] = (double)tk.AreaSum; break;
                    case "N": PF[p] = (double)tk.Count; break;
                    case "SX": PF[p] = (double)tk.Slope.X; break;
                    case "SY": PF[p] = (double)tk.Slope.Y; break;
                    case "S": PF[p] = (double)tk.Sigma; break;

                    default: throw new Exception("Unknown parameter " + PFParams[p]);
                }
            }
            return (PF.Evaluate() != 0.0);
        }

        internal class IntTrack
        {
            public uint AreaSum;
            public uint Count;
            public SySal.BasicTypes.Vector Intercept;
            public SySal.BasicTypes.Vector Slope;
            public double Sigma;
            public double TopZ, BottomZ;
            public bool Valid;
            public uint OriginalFragment;
            public int OriginalView;
            public int OriginalIndex;
            public SySal.BasicTypes.Vector OriginalFieldIntercept;
            public SySal.BasicTypes.Vector OriginalSlope;
            public int ViewId;
            public int ValidId;
            public SySal.Scanning.MIPIndexedEmulsionTrack NewTrack = null;
            public bool Linked;

            public IntTrack() { }

            public IntTrack(System.IO.BinaryReader b)
            {
                AreaSum = b.ReadUInt32();
                BottomZ = b.ReadDouble();
                Count = b.ReadUInt32();
                Intercept.X = b.ReadDouble();
                Intercept.Y = b.ReadDouble();
                Intercept.Z = b.ReadDouble();
                OriginalFieldIntercept.X = b.ReadDouble();
                OriginalFieldIntercept.Y = b.ReadDouble();
                OriginalFieldIntercept.Z = b.ReadDouble();
                OriginalFragment = b.ReadUInt32();
                OriginalIndex = b.ReadInt32();
                OriginalView = b.ReadInt32();
                ViewId = b.ReadInt32();
                Sigma = b.ReadDouble();
                Slope.X = b.ReadDouble();
                Slope.Y = b.ReadDouble();
                Slope.Z = b.ReadDouble();
                OriginalSlope.X = b.ReadDouble();
                OriginalSlope.Y = b.ReadDouble();
                OriginalSlope.Z = b.ReadDouble();
                TopZ = b.ReadDouble();
                Valid = b.ReadBoolean();
                Linked = b.ReadBoolean();
                ValidId = b.ReadInt32();
            }

            public void Save(System.IO.BinaryWriter b)
            {
                b.Write(AreaSum);
                b.Write(BottomZ);
                b.Write(Count);
                b.Write(Intercept.X);
                b.Write(Intercept.Y);
                b.Write(Intercept.Z);
                b.Write(OriginalFieldIntercept.X);
                b.Write(OriginalFieldIntercept.Y);
                b.Write(OriginalFieldIntercept.Z);
                b.Write(OriginalFragment);
                b.Write(OriginalIndex);
                b.Write(OriginalView);
                b.Write(ViewId);
                b.Write(Sigma);
                b.Write(Slope.X);
                b.Write(Slope.Y);
                b.Write(Slope.Z);
                b.Write(OriginalSlope.X);
                b.Write(OriginalSlope.Y);
                b.Write(OriginalSlope.Z);
                b.Write(TopZ);
                b.Write(Valid);
                b.Write(Linked);
                b.Write(ValidId);
            }

            public IntTrack(SySal.Tracking.MIPEmulsionTrack t, IntView owner, int originalindex, int viewid, SySal.BasicTypes.Vector ofi, SySal.BasicTypes.Vector os)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = t.Info;
                AreaSum = info.AreaSum;
                Count = info.Count;
                Intercept = info.Intercept;
                OriginalFieldIntercept = ofi;
                Slope = info.Slope;
                OriginalSlope = os;
                Sigma = info.Sigma;
                TopZ = info.TopZ;
                BottomZ = info.BottomZ;
                Valid = true;
                OriginalFragment = owner.Owner.Index;
                OriginalView = owner.OriginalIndex;
                OriginalIndex = originalindex;
                ViewId = viewid;
                Linked = false;
                ValidId = -1;
            }
        }

        internal class IntCell
        {
            public int Fill;
            public IntTrack[] Tracks;
        }

        internal class IntSide
        {
            private const int MaxCells = 10000;

            public int Fill;
            public IntTrack[] Tracks;
            public IntCell[,] Cells;
            public double MinX, MinY;
            public double ZBase, ZExt;
            public int Id;
            public double cellpostol;
            public SySal.BasicTypes.Vector2 MapPos;

            public IntSide(Fragment.View.Side s, bool istop, double postol, double minslope2, int id, IntView owner)
            {
                int i;
                Id = id;
                MapPos = s.MapPos;
                double TopZ = s.TopZ;
                double BottomZ = s.BottomZ;
                ZBase = istop ? BottomZ : TopZ;
                ZExt = istop ? TopZ : BottomZ;
                Tracks = new IntTrack[s.Length];                
                for (i = Fill = 0; i < Tracks.Length; i++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = s[i].Info;
                    IntTrack intk = new IntTrack(s[i], owner, i, Id, info.Intercept, info.Slope);
                    intk.Intercept.X = (double)(MapPos.X + s.MXX * info.Intercept.X + s.MXY * info.Intercept.Y);
                    intk.Intercept.Y = (double)(MapPos.Y + s.MYX * info.Intercept.X + s.MYY * info.Intercept.Y);
                    intk.Slope.X = (double)(s.MXX * info.Slope.X + s.MXY * info.Slope.Y);
                    intk.Slope.Y = (double)(s.MYX * info.Slope.X + s.MYY * info.Slope.Y);
                    Tracks[Fill++] = intk;
                }
                if (Fill > 0)
                {
                    double MaxX, MaxY;
                    MaxX = MinX = Tracks[0].Intercept.X;
                    MaxY = MinY = Tracks[0].Intercept.Y;
                    for (i = 1; i < Fill; i++)
                    {
                        if (Tracks[i].Intercept.X < MinX) MinX = Tracks[i].Intercept.X;
                        else if (Tracks[i].Intercept.X > MaxX) MaxX = Tracks[i].Intercept.X;
                        if (Tracks[i].Intercept.Y < MinY) MinY = Tracks[i].Intercept.Y;
                        else if (Tracks[i].Intercept.Y > MaxY) MaxY = Tracks[i].Intercept.Y;
                    }
                    int xc, yc, ix, iy;
                    cellpostol = postol;
                    try
                    {
                        while (true)
                        {
                            xc = (int)Math.Floor((MaxX - MinX) / cellpostol + 0.5f) + 3;
                            yc = (int)Math.Floor((MaxY - MinY) / cellpostol + 0.5f) + 3;
                            if (xc < 0 || yc < 0) throw new Exception();
                            if ((long)xc * (long)yc > (long)MaxCells) cellpostol *= 2.0;
                            else break;
                        }
                    }
                    catch (Exception)
                    {
                        xc = yc = 1;
                        Tracks = new IntTrack[Fill = 0];
                        MinX = MinY = 0.0;
                    }
                    MinX -= cellpostol;
                    MinY -= cellpostol;
                    Cells = new IntCell[yc, xc];
                    for (iy = 0; iy < yc; iy++)
                        for (ix = 0; ix < xc; ix++)
                            Cells[iy, ix] = new IntCell();
                    IntCell[] tempindex = new IntCell[Fill];
                    for (i = 0; i < Fill; i++)
                    {
                        ix = (int)Math.Floor((Tracks[i].Intercept.X - MinX) / cellpostol + 0.5);
                        iy = (int)Math.Floor((Tracks[i].Intercept.Y - MinY) / cellpostol + 0.5);
                        (tempindex[i] = Cells[iy, ix]).Fill++;
                    }
                    for (iy = 0; iy < yc; iy++)
                        for (ix = 0; ix < xc; ix++)
                        {
                            Cells[iy, ix].Tracks = new IntTrack[Cells[iy, ix].Fill];
                            Cells[iy, ix].Fill = 0;
                        }
                    for (i = 0; i < Fill; i++)
                        tempindex[i].Tracks[tempindex[i].Fill++] = Tracks[i];

                    i = 0;
                    for (iy = 0; iy < yc; iy++)
                        for (ix = 0; ix < xc; ix++)
                            foreach (IntTrack t in Cells[iy, ix].Tracks)
                                Tracks[i++] = t;
                }
                else
                {
                    MinX = MinY = 0;
                    Cells = new IntCell[0, 0];
                }
            }

            public void Clean(IntSide s, double postol, double mergepostol2, double mergeslopetol2, bool singlethread)
            {
                int threadindex;
                int processors = singlethread ? 1 : System.Environment.ProcessorCount;
                System.Threading.Thread[] threads = new System.Threading.Thread[processors];
                for (threadindex = 0; threadindex < threads.Length; threadindex++)
                {
                    threads[threadindex] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(
                        o =>
                        {
                            int thri = (int)o;
                            int i, j, ix, iy, iix, iiy, sx, sy;
                            double slx, sly, inx, iny;
                            sy = s.Cells.GetLength(0);
                            sx = s.Cells.GetLength(1);
                            for (i = thri; i < Fill; i += processors)
                            {
                                IntTrack pt = Tracks[i];
                                slx = pt.Slope.X;
                                sly = pt.Slope.Y;
                                ix = (int)(((inx = pt.Intercept.X) - s.MinX) / cellpostol + 0.5);
                                iy = (int)(((iny = pt.Intercept.Y) - s.MinY) / cellpostol + 0.5);
                                for (iiy = iy - 1; iiy <= iy + 1; iiy++)
                                    if (iiy >= 0 && iiy < sy)
                                        for (iix = ix - 1; iix <= ix + 1; iix++)
                                            if (iix >= 0 && iix < sx)
                                            {
                                                IntCell c = s.Cells[iiy, iix];
                                                for (j = 0; j < c.Fill; j++)
                                                {
                                                    IntTrack ptt = c.Tracks[j];
                                                    double dslx = slx - ptt.Slope.X;
                                                    double dsly = sly - ptt.Slope.Y;
                                                    if ((dslx * dslx + dsly * dsly) < mergeslopetol2 && !pt.Equals(ptt))
                                                    {
                                                        double dinx = inx - ptt.Intercept.X;
                                                        double diny = iny - ptt.Intercept.Y;
                                                        if ((dinx * dinx + diny * diny) < mergepostol2)
                                                        {
                                                            if (ptt.Linked) pt.Valid = false;
                                                            else if (pt.Linked) ptt.Valid = false;
                                                            else if (pt.Count > ptt.Count) ptt.Valid = false;
                                                            else if (pt.Count < ptt.Count) pt.Valid = false;
                                                            else if (pt.OriginalFragment < ptt.OriginalFragment ||
                                                                (pt.OriginalFragment == ptt.OriginalFragment && pt.OriginalView < ptt.OriginalView) ||
                                                                (pt.OriginalFragment == ptt.OriginalFragment && pt.OriginalView == ptt.OriginalView && pt.OriginalIndex < ptt.OriginalIndex))
                                                                ptt.Valid = false;
                                                            else pt.Valid = false;
                                                        }
                                                    }
                                                }
                                            }
                            }

                        }
                        ));
                    threads[threadindex].Start(threadindex);
                }
                for (threadindex = 0; threadindex < threads.Length; threadindex++) threads[threadindex].Join();
            }

            public ArrayList Link(IntSide s, int minpts, double minslope2, double postol, double slopetol, double slopetolincreasewithslope, dQualityCut [] qc, bool preventdup, int linklimit, SySal.DataStreams.OPERALinkedZone.Writer wr, bool keeplinkedtracksonly, ref int basecount, System.IO.StreamWriter logger = null)
            {
                SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry tie = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry();
                ArrayList a = new ArrayList(100);
                int threadindex;
                int processors = qc.Length;
                bool singlethread = processors == 1;                
                System.Threading.Thread[] threads = new System.Threading.Thread[processors];
                for (threadindex = 0; threadindex < threads.Length; threadindex++)
                {
                    threads[threadindex] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(
                        o =>
                            {
                int thri = (int)o;
                int i, j, ix, iy, iix, iiy, cy, cx;
                double FB, SB, DS;
                double FX, FY, SX, SY, NSX, NSY;
                double postol2 = postol * postol;
                double newslope, newslopetol;
                double dirx, diry;
                double dsx, dsy, dix, diy;
                double dds, dns;
                FB = ZBase;
                SB = s.ZBase;
                cy = s.Cells.GetLength(0);
                cx = s.Cells.GetLength(1);
                bool go_on;                
                for (i = thri; i < Fill; i += processors)
                {
                    IntTrack t = Tracks[i];                    
                    if (t.Valid && (!preventdup || !t.Linked) && (t.Count >= minpts) && ((t.Slope.X * t.Slope.X + t.Slope.Y * t.Slope.Y) > minslope2))
                    {
                        go_on = true;
                        FX = t.Intercept.X + (SB - t.Intercept.Z) * t.Slope.X;
                        FY = t.Intercept.Y + (SB - t.Intercept.Z) * t.Slope.Y;
                        ix = (int)((FX - s.MinX) / cellpostol + 0.5);
                        iy = (int)((FY - s.MinY) / cellpostol + 0.5);
                        for (iiy = iy - 1; go_on && iiy <= iy + 1; iiy++)
                            if (iiy >= 0 && iiy < cy)
                                for (iix = ix - 1; go_on && iix <= ix + 1; iix++)
                                    if (iix >= 0 && iix < cx)
                                    {
                                        IntCell c = s.Cells[iiy, iix];
                                        for (j = 0; j < c.Fill; j++)
                                        {
                                            IntTrack tt = c.Tracks[j];
                                            if (tt.Valid && (!preventdup || !tt.Linked) && (tt.Count >= minpts) && ((tt.Slope.X * tt.Slope.X + tt.Slope.Y * tt.Slope.Y) > minslope2))
                                            {
                                                NSX = (t.Intercept.X - tt.Intercept.X) / (t.Intercept.Z - tt.Intercept.Z);
                                                NSY = (t.Intercept.Y - tt.Intercept.Y) / (t.Intercept.Z - tt.Intercept.Z);
                                                if ((newslope = Math.Sqrt(NSX * NSX + NSY * NSY)) > 0.0)
                                                {
                                                    dirx = NSX / newslope;
                                                    diry = NSY / newslope;
                                                }
                                                else
                                                {
                                                    dirx = 1.0;
                                                    diry = 0.0;
                                                }
                                                newslopetol = slopetol + newslope * slopetolincreasewithslope;
                                                dsx = t.Slope.X - NSX;
                                                dsy = t.Slope.Y - NSY;
                                                if ((dns = Math.Abs(dirx * dsy - diry * dsx)) > slopetol) continue;
                                                if ((dds = Math.Abs(dirx * dsx + diry * dsy)) > newslopetol) continue;
                                                dns /= slopetol;
                                                dds /= newslopetol;
                                                DS = dns * dns + dds * dds;
                                                dsx = tt.Slope.X - NSX;
                                                dsy = tt.Slope.Y - NSY;
                                                if ((dns = Math.Abs(dirx * dsy - diry * dsx)) > slopetol) continue;
                                                if ((dds = Math.Abs(dirx * dsx + diry * dsy)) > newslopetol) continue;
                                                dns /= slopetol;
                                                dds /= newslopetol;
                                                DS += (dns * dns + dds * dds);
                                                dix = FX - tt.Intercept.X;
                                                diy = FY - tt.Intercept.Y;
                                                if ((dix * dix + diy * diy) > postol2) continue;
                                                SX = tt.Intercept.X + (FB - tt.Intercept.Z) * tt.Slope.X;
                                                SY = tt.Intercept.Y + (FB - tt.Intercept.Z) * tt.Slope.Y;
                                                dix = SX - t.Intercept.X;
                                                diy = SY - t.Intercept.Y;
                                                if ((dix * dix + diy * diy) > postol2) continue;
                                                SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();

                                                info.AreaSum = t.AreaSum + tt.AreaSum;
                                                info.Count = (ushort)(t.Count + tt.Count);
                                                info.Intercept = t.Intercept;
                                                info.Slope.X = (double)NSX;
                                                info.Slope.Y = (double)NSY;
                                                info.Slope.Z = 1.0f;
                                                info.TopZ = t.TopZ;
                                                info.BottomZ = tt.BottomZ;
                                                info.Sigma = (double)Math.Sqrt(DS);
                                                IntMIPIndexedBaseTrack b = new IntMIPIndexedBaseTrack(a.Count, info, null, null);

                                                b.IntTop = t;
                                                b.IntBottom = tt;
                                                if (qc[thri](b))
                                                {
                                                    lock (a)
                                                    {
                                                        a.Add(b);
                                                        if (linklimit > 0 && a.Count >= linklimit)
                                                        {
                                                            a.Clear();
                                                            return;
                                                        }
                                                    }
                                                    t.Linked = true;
                                                    tt.Linked = true;
                                                    if (logger != null)
                                                        lock (logger)
                                                        {
                                                            logger.WriteLine(
                                                                this.Id + " " + t.ViewId + " " + t.OriginalFragment + " " + t.OriginalView + " " + t.OriginalIndex + " " + t.Count + " " + t.AreaSum +
                                                                " " + t.Intercept.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.Intercept.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.Intercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.Sigma.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.TopZ.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.BottomZ.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + this.ZBase.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + this.ZExt.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.OriginalFieldIntercept.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.OriginalFieldIntercept.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.OriginalFieldIntercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.OriginalSlope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.OriginalSlope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + t.OriginalSlope.Z.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + " " +
                                                                s.Id + " " + tt.ViewId + " " + tt.OriginalFragment + " " + tt.OriginalView + " " + tt.OriginalIndex + " " + tt.Count + " " + tt.AreaSum +
                                                                " " + tt.Intercept.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.Intercept.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.Intercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.Sigma.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.TopZ.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.BottomZ.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + s.ZBase.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + s.ZExt.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.OriginalFieldIntercept.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.OriginalFieldIntercept.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.OriginalFieldIntercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.OriginalSlope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.OriginalSlope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + tt.OriginalSlope.Z.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + b.Info.Count + " " + b.Info.AreaSum +
                                                                " " + b.Info.Intercept.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + b.Info.Intercept.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + b.Info.Intercept.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + b.Info.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + b.Info.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) +
                                                                " " + b.Info.Sigma.ToString("F5", System.Globalization.CultureInfo.InvariantCulture)
                                                                );
                                                        }
                                                    if (preventdup)
                                                    {
                                                        go_on = false;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                    }
                }
                            }
                    ));
                    threads[threadindex].Start(threadindex);
                }
                for (threadindex = 0; threadindex < threads.Length; threadindex++) threads[threadindex].Join();
                if (wr != null)
                    foreach (IntMIPIndexedBaseTrack btk in a)
                    {
                        if (keeplinkedtracksonly)
                        {
                            wr.AddMIPBasetrack(btk.Info, basecount, basecount, basecount);
                            IntTrack itk;
                            itk = btk.IntTop;
                            SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                            info.Field = 0;
                            info.Count = (ushort)itk.Count;
                            info.AreaSum = itk.AreaSum;
                            info.Intercept = itk.Intercept;
                            info.Slope = itk.Slope;
                            info.Sigma = itk.Sigma;
                            info.TopZ = itk.TopZ;
                            info.BottomZ = itk.BottomZ;
                            tie.Fragment = (int)itk.OriginalFragment;
                            tie.View = itk.OriginalView;
                            tie.Track = itk.OriginalIndex;
                            wr.AddMIPEmulsionTrack(info, basecount, itk.ViewId, tie, true);
                            itk = btk.IntBottom;
                            info.Field = 0;
                            info.Count = (ushort)itk.Count;
                            info.AreaSum = itk.AreaSum;
                            info.Intercept = itk.Intercept;
                            info.Slope = itk.Slope;
                            info.Sigma = itk.Sigma;
                            info.TopZ = itk.TopZ;
                            info.BottomZ = itk.BottomZ;
                            tie.Fragment = (int)itk.OriginalFragment;
                            tie.View = itk.OriginalView;
                            tie.Track = itk.OriginalIndex;
                            wr.AddMIPEmulsionTrack(info, basecount, itk.ViewId, tie, false);
                        }
                        else wr.AddMIPBasetrack(btk.Info, basecount, btk.IntTop.ValidId, btk.IntBottom.ValidId);
                        basecount++;
                    }
                return a;
            }

            public ArrayList PromoteMicrotracks(dPromotion pc, double OtherBase, bool IsTop, int linklimit, SySal.DataStreams.OPERALinkedZone.Writer wr, ref int basecount)
            {
                SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry tie = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry();
                ArrayList a = new ArrayList(100);
                foreach (IntTrack t in Tracks)
                    if ((t.Linked == false) && t.Valid && pc(t))
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                        IntTrack o = new IntTrack();

                        info.AreaSum = t.AreaSum;
                        info.Count = (ushort)(t.Count);
                        info.Intercept = t.Intercept;
                        info.Slope.X = t.Slope.X;
                        info.Slope.Y = t.Slope.Y;
                        info.Slope.Z = 1.0f;
                        if (IsTop)
                        {
                            info.TopZ = t.TopZ;
                            info.BottomZ = t.BottomZ;//info.BottomZ = OtherBase;
                        }
                        else
                        {
                            info.BottomZ = t.BottomZ;
                            info.TopZ = t.TopZ; //info.TopZ = OtherBase;
                            info.Intercept.X += info.Slope.X * (OtherBase - info.Intercept.Z);
                            info.Intercept.Y += info.Slope.Y * (OtherBase - info.Intercept.Z);
                            info.Intercept.Z = OtherBase;
                        }
                        info.Sigma = -1.0;

                        o.AreaSum = 0;
                        o.Count = 0;
                        o.Intercept.X = t.Intercept.X + (OtherBase - t.Intercept.Z) * t.Slope.X;
                        o.Intercept.Y = t.Intercept.Y + (OtherBase - t.Intercept.Z) * t.Slope.Y;
                        o.Slope.X = t.Slope.X;
                        o.Slope.Y = t.Slope.Y;
                        o.Slope.Z = 1.0;
                        o.Intercept.Z = o.TopZ = o.BottomZ = OtherBase;
                        o.Sigma = -1.0;
                        o.OriginalView = t.ViewId;
                        o.OriginalFragment = t.OriginalFragment;
                        o.OriginalIndex = -1;
                        o.ViewId = t.ViewId;
                        o.Valid = true;
                        o.Linked = true;
                        IntMIPIndexedBaseTrack b = new IntMIPIndexedBaseTrack(a.Count, info, null, null);

                        b.IntTop = IsTop ? t : o;
                        b.IntBottom = IsTop ? o : t;
                        a.Add(b);
                        if (linklimit > 0 && a.Count >= linklimit)
                        {
                            a.Clear();
                            return a;
                        }
                    }
                if (wr != null)
                    foreach (IntMIPIndexedBaseTrack btk in a)
                    {
                        wr.AddMIPBasetrack(btk.Info, basecount, basecount, basecount);
                        IntTrack itk;
                        itk = btk.IntTop;
                        SySal.Tracking.MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
                        info.Field = 0;
                        info.Count = (ushort)itk.Count;
                        info.AreaSum = itk.AreaSum;
                        info.Intercept = itk.Intercept;
                        info.Slope = itk.Slope;
                        info.Sigma = itk.Sigma;
                        info.TopZ = itk.TopZ;
                        info.BottomZ = itk.BottomZ;
                        tie.Fragment = (int)itk.OriginalFragment;
                        tie.View = itk.OriginalView;
                        tie.Track = itk.OriginalIndex;
                        wr.AddMIPEmulsionTrack(info, basecount, itk.ViewId, tie, true);
                        itk = btk.IntBottom;
                        info.Field = 0;
                        info.Count = (ushort)itk.Count;
                        info.AreaSum = itk.AreaSum;
                        info.Intercept = itk.Intercept;
                        info.Slope = itk.Slope;
                        info.Sigma = itk.Sigma;
                        info.TopZ = itk.TopZ;
                        info.BottomZ = itk.BottomZ;
                        tie.Fragment = (int)itk.OriginalFragment;
                        tie.View = itk.OriginalView;
                        tie.Track = itk.OriginalIndex;
                        wr.AddMIPEmulsionTrack(info, basecount, itk.ViewId, tie, false);
                        basecount++;
                    }
                return a;
            }
        }

        internal class IntView : IDisposable
        {
            public int TileX, TileY;
            public IntSide Top, Bottom;
            public IntFragment Owner;
            public int OriginalIndex;
            public bool IsOnDisk;
            private string Path;
            private System.IO.FileStream SwapFile;

            static System.Random Rnd = new System.Random();

            internal IntView(Fragment.View v, double postol, double minslope2, IntFragment owner, int originalindex, int XSize, OPERALinkedZone.OPERAView[][] viewrecords)
            {
                Owner = owner;
                OriginalIndex = originalindex;
                TileX = v.Tile.X;
                TileY = v.Tile.Y;
                int Id = 0;
                if (viewrecords != null)
                {
                    Id = TileY * XSize + TileX;
                    OPERALinkedZone.OPERAView vw;
                    vw = new OPERALinkedZone.OPERAView(Id);
                    vw.SetInfo(v.Top.MapPos, v.Top.TopZ, v.Top.BottomZ);
                    viewrecords[0][Id] = vw;
                    vw = new OPERALinkedZone.OPERAView(Id);
                    vw.SetInfo(v.Bottom.MapPos, v.Bottom.TopZ, v.Bottom.BottomZ);
                    viewrecords[1][Id] = vw;
                }
                Top = new IntSide(v.Top, true, postol, minslope2, Id, this);
                Bottom = new IntSide(v.Bottom, false, postol, minslope2, Id, this);
                Path = Environment.ExpandEnvironmentVariables("%TEMP%\\stripesfraglink2_i_" + System.DateTime.Now.Ticks.ToString("X16") + Rnd.Next().ToString("X8") + "_v_" + TileX.ToString() + "_" + TileY.ToString() + ".tmp");
                SwapFile = null;
                IsOnDisk = false;
            }

            public void Clean(IntView w, double postol, double mergepostol2, double mergeslopetol2, bool singlethread)
            {
                if (IsOnDisk) ReloadFromDisk();
                if (w.IsOnDisk) w.ReloadFromDisk();
                Top.Clean(w.Top, postol, mergepostol2, mergeslopetol2, singlethread);
                Bottom.Clean(w.Bottom, postol, mergepostol2, mergeslopetol2, singlethread);
            }

            public ArrayList Link(IntView w, int minpts, double minslope2, double postol, double slopetol, double slopetolincreasewithslope, dQualityCut [] qc, bool preventdup, int linklimit, SySal.DataStreams.OPERALinkedZone.Writer wr, bool keeplinkedtracksonly, ref int basecount, System.IO.StreamWriter logger)
            {
                if (IsOnDisk) ReloadFromDisk();
                if (w.IsOnDisk) w.ReloadFromDisk();
                ArrayList n = Top.Link(w.Bottom, minpts, minslope2, postol, slopetol, slopetolincreasewithslope, qc, preventdup, linklimit, wr, keeplinkedtracksonly, ref basecount, logger);
                if (this != w) n.AddRange(w.Top.Link(Bottom, minpts, minslope2, postol, slopetol, slopetolincreasewithslope, qc, preventdup, linklimit, wr, keeplinkedtracksonly, ref basecount, logger));
                return n;
            }

            public ArrayList PromoteMicrotracks(dPromotion pc, SySal.DataStreams.OPERALinkedZone.Writer wr, int linklimit, ref int basecount)
            {
                if (IsOnDisk) ReloadFromDisk();
                ArrayList n = Top.PromoteMicrotracks(pc, Bottom.ZBase, true, linklimit, wr, ref basecount);
                n.AddRange(Bottom.PromoteMicrotracks(pc, Top.ZBase, false, linklimit, wr, ref basecount));
                return n;
            }

            public void MoveToDisk()
            {
                if (IsOnDisk) return;
                if (SwapFile == null) SwapFile = new System.IO.FileStream(Path, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                else SwapFile.Position = 0;
                System.IO.BinaryWriter b = new System.IO.BinaryWriter(SwapFile);
                int iy, ix, yc, xc;
                IntSide side = Top;
                do
                {
                    b.Write(yc = side.Cells.GetLength(0)); b.Write(xc = side.Cells.GetLength(1));
                    for (iy = 0; iy < yc; iy++)
                        for (ix = 0; ix < xc; ix++)
                        {
                            b.Write(side.Cells[iy, ix].Fill);
                            foreach (IntTrack t in side.Cells[iy, ix].Tracks)
                                t.Save(b);
                        }
                    side = (side == Top) ? (side = Bottom) : side = null;
                }
                while (side != null);
                b.Flush();
                Top.Cells = Bottom.Cells = null;
                Top.Tracks = Bottom.Tracks = null;
                IsOnDisk = true;
            }

            public void ReloadFromDisk()
            {
                if (!IsOnDisk) return;
                SwapFile.Position = 0;
                System.IO.BinaryReader b = new System.IO.BinaryReader(SwapFile);
                int iy, ix, yc, xc, nt, i, j;
                IntSide side = Top;
                do
                {
                    yc = b.ReadInt32(); xc = b.ReadInt32();
                    side.Cells = new IntCell[yc, xc];
                    i = 0;
                    for (iy = 0; iy < yc; iy++)
                        for (ix = 0; ix < xc; ix++)
                        {
                            IntCell c = side.Cells[iy, ix] = new IntCell();
                            i += (nt = c.Fill = b.ReadInt32());
                            c.Tracks = new IntTrack[nt];
                            for (j = 0; j < nt; j++)
                                c.Tracks[j] = new IntTrack(b);
                        }
                    side.Tracks = new IntTrack[i];
                    i = 0;
                    for (iy = 0; iy < yc; iy++)
                        for (ix = 0; ix < xc; ix++)
                            foreach (IntTrack t in side.Cells[iy, ix].Tracks)
                                side.Tracks[i++] = t;

                    side = (side == Top) ? (side = Bottom) : side = null;
                }
                while (side != null);
                IsOnDisk = false;
            }

            private void Dispose(bool isfinalizing)
            {
                try
                {
                    if (SwapFile != null) SwapFile.Close();
                    System.IO.File.Delete(Path);
                    SwapFile = null;
                }
                catch (Exception) { }
                if (Top != null)
                {
                    Top.Cells = null;
                    Top.Tracks = null;
                    Top = null;
                }
                if (Bottom != null)
                {
                    Bottom.Cells = null;
                    Bottom.Tracks = null;
                    Bottom = null;
                }
                if (!isfinalizing) GC.SuppressFinalize(this);
            }

            public void Dispose()
            {
                Dispose(false);
            }

            ~IntView()
            {
                Dispose(true);
            }
        }

        internal class IntFragment
        {
            public uint Index;
            public IntView[] Views;

            public IntFragment(Fragment f, int TileLine, double postol, double minslope2, bool movetodisk, int XSize, OPERALinkedZone.OPERAView[][] viewrecords)
            {
                int i, csize;

                csize = 0;
                Index = f.Index;
                for (i = 0; i < f.Length; i++)
                    if (f[i].Tile.Y == TileLine) csize++;
                Views = new IntView[csize];
                if (csize == 0) return;
                csize = 0;
                for (i = 0; i < f.Length; i++)
                    if (f[i].Tile.Y == TileLine)
                    {
                        Views[csize] = new IntView(f[i], postol, minslope2, this, i, XSize, viewrecords);
                        if (movetodisk) Views[csize].MoveToDisk();
                        csize++;
                    }
            }
        }

        class IntTileLine
        {
            public int Fill;
            public IntFragment[] Fragments;
        }

        void LoadFragment(uint index, bool[] FragmentLoaded, IntTileLine[] TileLines, int XSize, OPERALinkedZone.OPERAView[][] viewrecords, ref bool TransformSet, ref SySal.DAQSystem.Scanning.IntercalibrationInfo TransformInfo)
        {
            if (!FragmentLoaded[index - 1])
            {
                Fragment Frag = intLoad(index);
                if (!TransformSet && Frag.Length > 0)
                {
                    TransformInfo.MXX = Frag[0].Top.MXX;
                    TransformInfo.MXY = Frag[0].Top.MXY;
                    TransformInfo.MYX = Frag[0].Top.MYX;
                    TransformInfo.MYY = Frag[0].Top.MYY;
                    TransformInfo.TX = Frag[0].Top.MapPos.X - (Frag[0].Top.MXX * Frag[0].Top.Pos.X + Frag[0].Top.MXY * Frag[0].Top.Pos.Y);
                    TransformInfo.TY = Frag[0].Top.MapPos.Y - (Frag[0].Top.MYX * Frag[0].Top.Pos.X + Frag[0].Top.MYY * Frag[0].Top.Pos.Y);
                    TransformInfo.RX = 0.0;
                    TransformInfo.RY = 0.0;
                    TransformSet = true;
                }
                FragmentLoaded[index - 1] = true;
                if (Frag.Length > 0)
                {
                    int miny, maxy, i;

                    maxy = miny = Frag[0].Tile.Y;
                    for (i = 1; i < Frag.Length; i++)
                    {
                        if (Frag[i].Tile.Y > maxy) maxy = Frag[i].Tile.Y;
                        else if (Frag[i].Tile.Y < miny) miny = Frag[i].Tile.Y;
                    }
                    if (maxy >= TileLines.Length) maxy = TileLines.Length - 1;
                    for (i = miny; i <= maxy; i++)
                    {
                        IntTileLine t;
                        if (TileLines[i] == null)
                        {
                            t = (TileLines[i] = new IntTileLine());
                            t.Fragments = new IntFragment[XSize];
                        }
                        else t = TileLines[i];
                        t.Fragments[t.Fill++] = new IntFragment(Frag, i, C.PosTol, C.MinSlope * C.MinSlope, C.KeepLinkedTracksOnly && C.MemorySaving >= 1, XSize, viewrecords);
                    }
                }
            }
        }
        #endregion

        #region ILinkProcessor

        /// <summary>
        /// Links RWD files into a LinkedZone. The output format is left to StripesFragLink2. Currently, <c>SySal.Scanning.Plate.IO.OPERA.LinkedZone</c> format LinkedZones are produced.
        /// </summary>
        /// <param name="Cat">the Catalog of the Raw Data Files.</param>
        /// <returns>the LinkedZone produced.</returns>
        public SySal.Scanning.Plate.IO.OPERA.LinkedZone Link(Catalog Cat) { return (SySal.Scanning.Plate.IO.OPERA.LinkedZone)Link(Cat, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone)); }

        #endregion
    }
}
