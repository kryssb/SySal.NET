using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.DataStreams
{
    /// <summary>
    /// Allows accessing LinkedZone data from TLG files without loading the full information in memory.
    /// </summary>
    public class OPERALinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone, IDisposable
    {        
        /// <summary>
        /// File-resident MIPBaseTrack.
        /// </summary>
        protected class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
        {
            uint m_IdTop;
            uint m_IdBottom;
            Side m_TopSide;
            Side m_BottomSide;

            /// <summary>
            /// Reads a MIPBaseTrack from file, along with its associated microtracks.
            /// </summary>
            /// <param name="t">the top Side of the LinkedZone.</param>
            /// <param name="b">the bottom Side of the LinkedZone.</param>
            /// <param name="id">the Id of the track to be read.</param>
            /// <param name="r">the <c>BinaryReader</c> wrapping the data containing file.</param>
            public MIPBaseTrack(Side t, Side b, int id, System.IO.BinaryReader r)
            {
                m_TopSide = t;
                m_BottomSide = b;
                m_Id = id;
                SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                info.AreaSum = (ushort)r.ReadUInt32();
                info.Count = (ushort)r.ReadUInt32();
                info.Intercept.X = r.ReadDouble();
                info.Intercept.Y = r.ReadDouble();
                info.Intercept.Z = r.ReadDouble();
                info.Slope.X = r.ReadDouble();
                info.Slope.Y = r.ReadDouble();
                info.Slope.Z = r.ReadDouble();
                info.Sigma = r.ReadDouble();                
                m_IdTop = r.ReadUInt32();
                m_IdBottom = r.ReadUInt32();
                info.TopZ = t[(int)m_IdTop].Info.TopZ;
                info.BottomZ = b[(int)m_IdBottom].Info.BottomZ;
                m_Info = info;
            }

            /// <summary>
            /// Yields the top microtrack of the MIPBaseTrack.
            /// </summary>
            public override SySal.Scanning.MIPIndexedEmulsionTrack Top { get { return m_TopSide[(int)m_IdTop]; } }

            /// <summary>
            /// Yields the bottom microtrack of the MIPBaseTrack.
            /// </summary>
            public override SySal.Scanning.MIPIndexedEmulsionTrack Bottom { get { return m_BottomSide[(int)m_IdBottom]; } }

            /// <summary>
            /// The size of the MIPBaseTrack in bytes.
            /// </summary>
            public const long Size = 72;
        }

        /// <summary>
        /// The last MIPBaseTrack loaded. Used to quickly retrieve recently used information without disk access.
        /// </summary>
        protected MIPBaseTrack m_LastTrack = null;

        /// <summary>
        /// File-resident MIPIndexedEmulsionTrack (i.e. a microtrack).
        /// </summary>
        protected new class MIPIndexedEmulsionTrack : SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack
        {
            int m_IdView;

            Side m_Side;

            /// <summary>
            /// Reads a microtrack from disk.
            /// </summary>
            /// <param name="s">the Side that contains the microtrack.</param>
            /// <param name="id">the Id of the microtrack to be accessed.</param>
            /// <param name="r">the <c>BinaryReader</c> wrapping the data containing file.</param>
            public MIPIndexedEmulsionTrack(Side s, int id, System.IO.BinaryReader r)
            {
                m_Side = s;
                SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                info.Field = r.ReadUInt32();
                info.AreaSum = r.ReadUInt32();
                info.Count = (ushort)r.ReadUInt32();
                info.Intercept.X = r.ReadDouble();
                info.Intercept.Y = r.ReadDouble();
                info.Intercept.Z = r.ReadDouble();
                info.Slope.X = r.ReadDouble();
                info.Slope.Y = r.ReadDouble();
                info.Slope.Z = r.ReadDouble();
                info.Sigma = r.ReadDouble();
                info.TopZ = r.ReadDouble();
                info.BottomZ = r.ReadDouble();
                m_OriginalRawData.Fragment = 0;
                m_Info = info;
                m_Id = id;
                m_IdView = r.ReadInt32();
            }

            /// <summary>
            /// Reads the View containing the microtrack.
            /// </summary>
            public override SySal.Scanning.Plate.IO.OPERA.LinkedZone.View View
            {
                get { return m_Side.View(m_IdView); }
            }

            /// <summary>
            /// Reads index information pointing to the original raw data files.
            /// </summary>
            public override SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry OriginalRawData
            {
                get 
                {
                    if (m_OriginalRawData.Fragment == 0)                    
                        m_Side.IndexEntry(m_Id, ref m_OriginalRawData);
                    return m_OriginalRawData; 
                }
            }

            /// <summary>
            /// The size of the MIPIndexedEmulsionTrack in bytes.
            /// </summary>
            public const long Size = 88;

            /// <summary>
            /// The size of each Index information entry.
            /// </summary>
            public const long IndexSize = 12;
        }

        /// <summary>
        /// File-persistent View.
        /// </summary>
        protected new class View : SySal.Scanning.Plate.IO.OPERA.LinkedZone.View
        {
            /// <summary>
            /// Reads a view from file.
            /// </summary>
            /// <param name="s">the Side containing the view.</param>
            /// <param name="r">the <c>BinaryReader</c> wrapping the data containing file.</param>
            public View(Side s, System.IO.BinaryReader r)
            {
                m_Side = s;
                m_Id = r.ReadInt32();
                m_Position.X = r.ReadDouble();
                m_Position.Y = r.ReadDouble();
                m_TopZ = r.ReadDouble();
                m_BottomZ = r.ReadDouble();
            }

            public const long Size = 36;
        }

        /// <summary>
        /// File-persistent Side.
        /// </summary>
        protected new class Side : SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side
        {
            System.IO.BinaryReader r;
            System.IO.Stream s;

            int N_Views;
            long View_SP;
            int N_Tracks;
            long Track_SP;
            long TrackIndex_SP;

            /// <summary>
            /// Prepares a side for coming access requests.
            /// </summary>
            /// <param name="topz">the Z of the top edge of the side.</param>
            /// <param name="bottomz">the Z of the bottom edge of the side.</param>
            /// <param name="nviews">the number of views in the side.</param>
            /// <param name="view_sp">the file position where View information begins.</param>
            /// <param name="ntracks">the number of tracks in the side.</param>
            /// <param name="track_sp">the file position where Track information begins.</param>
            /// <param name="index_sp">the file position where Index information begins.</param>
            /// <param name="strm">the data containing stream.</param>
            public Side(double topz, double bottomz, int nviews, long view_sp, int ntracks, long track_sp, long index_sp, System.IO.Stream strm)
            {
                r = new System.IO.BinaryReader(s = strm);
                this.m_TopZ = topz;
                this.m_BottomZ = bottomz;
                N_Views = nviews;
                View_SP = view_sp;
                N_Tracks = ntracks;
                Track_SP = track_sp;
                TrackIndex_SP = index_sp;
            }

            /// <summary>
            /// Last view loaded. Used to quickly retrieve recently used information without disk access.
            /// </summary>
            protected View m_LastView = null;

            /// <summary>
            /// Reads a View.
            /// </summary>
            /// <param name="id">the Id of the view to be read.</param>
            /// <returns>the requested view.</returns>
            public override SySal.Scanning.Plate.IO.OPERA.LinkedZone.View View(int id)
            {
                if (m_LastView != null && m_LastView.Id == id) return m_LastView;
                s.Seek(View_SP + id * SySal.DataStreams.OPERALinkedZone.View.Size, System.IO.SeekOrigin.Begin);
                return m_LastView = new View(this, r);
            }

            /// <summary>
            /// The number of views in the Side.
            /// </summary>
            public override int ViewCount
            {
                get { return N_Views; }
            }

            /// <summary>
            /// Last microtrack loaded. Used to quickly retrieve recently used information without disk access.
            /// </summary>
            protected MIPIndexedEmulsionTrack m_LastTrack = null;

            /// <summary>
            /// The number of microtracks in the side.
            /// </summary>
            public override int Length { get { return N_Tracks; } }

            /// <summary>
            /// Allows accessing microtracks in an array-like fashion.
            /// </summary>
            /// <param name="index">the Id of the track to be retrieved.</param>
            /// <returns>the requested microtrack.</returns>
            public override SySal.Scanning.MIPIndexedEmulsionTrack this[int index]
            {
                get
                {
                    if (m_LastTrack != null && m_LastTrack.Id == index) return m_LastTrack;
                    s.Seek(Track_SP + MIPIndexedEmulsionTrack.Size * index, System.IO.SeekOrigin.Begin);
                    return m_LastTrack = new MIPIndexedEmulsionTrack(this, index, r);
                }
            }

            internal void IndexEntry(int id, ref SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry tie)
            {
                s.Seek(TrackIndex_SP + id * MIPIndexedEmulsionTrack.IndexSize, System.IO.SeekOrigin.Begin);
                tie.Fragment = r.ReadInt32();
                tie.View = r.ReadInt32();
                tie.Track = r.ReadInt32();
            }
        }

        uint N_Tracks;
        long Tracks_SP;

        long TrackIndex_SP;

        /// <summary>
        /// The number of MIPBaseTracks in the LinkedZone.
        /// </summary>
        public override int Length { get { return (int)N_Tracks; } }

        /// <summary>
        /// Allows accessing base tracks in an array-like fashion.
        /// </summary>
        /// <param name="index">the Id of the track to be retrieved.</param>
        /// <returns>the requested microtrack.</returns>
        public override SySal.Scanning.MIPBaseTrack this[int index]
        {
            get
            {
                if (m_LastTrack != null && m_LastTrack.Id == index) return m_LastTrack;
                m_Stream.Seek(Tracks_SP + MIPBaseTrack.Size * index, System.IO.SeekOrigin.Begin);
                return m_LastTrack = new MIPBaseTrack((Side)m_Top, (Side)m_Bottom, index, r);
            }
        }

        System.IO.Stream m_Stream;
        System.IO.BinaryReader r;
        bool m_AutoClose = false;

        ~OPERALinkedZone()
        {
            if (m_Stream != null && m_AutoClose) m_Stream.Close();
        }

        /// <summary>
        /// Opens a file for data retrieval.
        /// </summary>
        /// <param name="filepath">the path of the file to be opened.</param>
        /// <remarks> The file must exist prior to the call, and is open in <c>Read/Share Read mode</c>. It is closed when <see cref="Dispose"/> is called, or on object finalization.</remarks>
        public OPERALinkedZone(string filepath)
        {
            Read(new System.IO.FileStream(filepath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read));
            m_AutoClose = true;
        }

        /// <summary>
        /// Associates the LinkedZone to an existing stream.
        /// </summary>
        /// <param name="str">the stream from which information has to be retrieved.</param>
        /// <remarks> The stream is not closed when the object is finalized. The stream will not be reset at its beginning.</remarks>
        public OPERALinkedZone(System.IO.Stream str)
        {
            Read(str);
        }

        /// <summary>
        /// Reads summary information from a stream, preparing the LinkedZone and its Sides for coming access requests.
        /// </summary>
        /// <param name="str">the stream from which data have to be retrieved.</param>
        protected void Read(System.IO.Stream str)
        {
            r = new System.IO.BinaryReader(m_Stream = str);
            byte infotype = r.ReadByte();
            ushort headerformat = r.ReadUInt16();
            if (infotype == ((byte)0x41))
                switch (headerformat)
                {
                    case (ushort)0x07: break;

                    default: throw new SystemException("Unsupported format");
                };

            if (r.ReadByte() != SectionTag) throw new Exception("The first section in a TLG file must contain tracks!");
            r.ReadInt64();

            m_Transform.MXX = m_Transform.MYY = 1.0;
            m_Transform.MXY = m_Transform.MYX = 0.0;
            m_Transform.TX = m_Transform.TY = 0.0;
            m_Transform.RX = m_Transform.RY = 0.0;

            m_Id.Part0 = r.ReadInt32();
            m_Id.Part1 = r.ReadInt32();
            m_Id.Part2 = r.ReadInt32();
            m_Id.Part3 = r.ReadInt32();

            m_Center.X = r.ReadDouble();
            m_Center.Y = r.ReadDouble();
            m_Extents.MinX = r.ReadDouble();
            m_Extents.MaxX = r.ReadDouble();
            m_Extents.MinY = r.ReadDouble();
            m_Extents.MaxY = r.ReadDouble();
            m_Transform.MXX = r.ReadDouble();
            m_Transform.MXY = r.ReadDouble();
            m_Transform.MYX = r.ReadDouble();
            m_Transform.MYY = r.ReadDouble();
            m_Transform.TX = r.ReadDouble();
            m_Transform.TY = r.ReadDouble();
            m_Transform.RX = r.ReadDouble();
            m_Transform.RY = r.ReadDouble();

            int ntopviews = r.ReadInt32();
            int nbottomviews = r.ReadInt32();
            double toptz = r.ReadDouble();
            double topbz = r.ReadDouble();
            double bottomtz = r.ReadDouble();
            double bottombz = r.ReadDouble();
            long topviews_sp = str.Position;
            str.Seek(ntopviews * View.Size, System.IO.SeekOrigin.Current);
            long bottomviews_sp = str.Position;
            str.Seek(nbottomviews * View.Size, System.IO.SeekOrigin.Current);
            uint ntoptracks = r.ReadUInt32();
            uint nbottomtracks = r.ReadUInt32();
            N_Tracks = r.ReadUInt32();
            long toptracks_sp = str.Position;
            long bottomtracks_sp = toptracks_sp + MIPIndexedEmulsionTrack.Size * ntoptracks;
            Tracks_SP = bottomtracks_sp + MIPIndexedEmulsionTrack.Size * nbottomtracks;
            long toptrackindices_sp = Tracks_SP + MIPBaseTrack.Size * N_Tracks;
            long bottomtrackindices_sp = toptrackindices_sp + MIPIndexedEmulsionTrack.IndexSize * ntoptracks;

            m_Top = new Side(toptz, topbz, (int)ntopviews, topviews_sp, (int)ntoptracks, toptracks_sp, toptrackindices_sp, str);
            m_Bottom = new Side(bottomtz, bottombz, (int)nbottomviews, bottomviews_sp, (int)nbottomtracks, bottomtracks_sp, bottomtrackindices_sp, str);
        }

        /// <summary>
        /// Creates a <see cref="SySal.Scanning.Plate.IO.OPERA.LinkedZone"/> from an open stream.
        /// </summary>
        /// <param name="strm">the stream from which data have to be retrieved.</param>
        /// <returns>the requested LinkedZone</returns>
        /// <remarks>If the file format is "MultiSection" or higher, a <see cref="SySal.DataStream.OPERALinkedZone"/> will be generated (so minimu memory occupancy is 
        /// achieved); in case this fails, a standard LinkedZone, loaded in memory, will be produced.</remarks>
        public static SySal.Scanning.Plate.IO.OPERA.LinkedZone FromStream(System.IO.Stream strm)
        {
            long startpos = strm.Position;
            try
            {
                return new OPERALinkedZone(strm);
            }
            catch (Exception)
            {
                strm.Seek(startpos, System.IO.SeekOrigin.Begin);
                return new SySal.Scanning.Plate.IO.OPERA.LinkedZone(strm);
            }
        }

        /// <summary>
        /// Creates a <see cref="SySal.Scanning.Plate.IO.OPERA.LinkedZone"/> from a file.
        /// </summary>
        /// <param name="filepath">the path of the file from which data have to be retrieved.</param>
        /// <returns>the requested LinkedZone</returns>
        /// <remarks>If the file format is "MultiSection" or higher, a <see cref="SySal.DataStream.OPERALinkedZone"/> will be generated (so minimu memory occupancy is 
        /// achieved); in case this fails, a standard LinkedZone, loaded in memory, will be produced.</remarks>
        public static SySal.Scanning.Plate.IO.OPERA.LinkedZone FromFile(string filepath)
        {
            try
            {
                return new OPERALinkedZone(filepath);
            }
            catch (Exception)
            {
                System.IO.FileStream strm = null;
                try
                {
                    strm = new System.IO.FileStream(filepath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read);
                    return new SySal.Scanning.Plate.IO.OPERA.LinkedZone(strm);
                }
                finally
                {
                    if (strm != null)
                    {
                        strm.Close();
                        strm = null;
                    }
                }
            }
        }

        #region IDisposable Members

        /// <summary>
        /// Releases all resources and unlocks the underlying stream/file.
        /// </summary>
        public void Dispose()
        {
            if (m_Stream != null && m_AutoClose)
            {
                m_Stream.Close();
                m_Stream = null;
            }
            GC.SuppressFinalize(this);
        }

        #endregion

        /// <summary>
        /// Writes a TLG in incremental mode, without need to host it in memory.
        /// </summary>
        public class Writer : IDisposable
        {
            string guid;
            string m_FilePath;
            System.IO.BinaryWriter w_tlg;
            System.IO.FileStream t_toptk;
            System.IO.FileStream t_topvw;
            System.IO.FileStream t_topix;
            System.IO.FileStream t_bottk;
            System.IO.FileStream t_botvw;
            System.IO.FileStream t_botix;
            System.IO.FileStream t_linked;
            System.IO.BinaryWriter b_toptk;
            System.IO.BinaryWriter b_topvw;
            System.IO.BinaryWriter b_topix;
            System.IO.BinaryWriter b_bottk;
            System.IO.BinaryWriter b_botvw;
            System.IO.BinaryWriter b_botix;
            System.IO.BinaryWriter b_linked;

            int toptks = 0;
            int bottks = 0;
            int linked = 0;
            int topvws = 0;
            int botvws = 0;

            long Section_Tracks_pos = 0;

            public Writer(string filepath, SySal.BasicTypes.Identifier id, SySal.BasicTypes.Rectangle extents, SySal.BasicTypes.Vector2 center, SySal.DAQSystem.Scanning.IntercalibrationInfo transform)
            {
                guid = System.Environment.ExpandEnvironmentVariables("%TEMP%\\" + System.Guid.NewGuid().ToString() + ".tlgs.");
                m_FilePath = filepath;
                w_tlg = new System.IO.BinaryWriter(new System.IO.FileStream(filepath, System.IO.FileMode.Create, System.IO.FileAccess.Write, System.IO.FileShare.None));
                b_toptk = new System.IO.BinaryWriter(t_toptk = new System.IO.FileStream(guid + "ttk", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));
                b_bottk = new System.IO.BinaryWriter(t_bottk = new System.IO.FileStream(guid + "btk", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));
                b_topvw = new System.IO.BinaryWriter(t_topvw = new System.IO.FileStream(guid + "tvw", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));
                b_botvw = new System.IO.BinaryWriter(t_botvw = new System.IO.FileStream(guid + "bvw", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));
                b_topix = new System.IO.BinaryWriter(t_topix = new System.IO.FileStream(guid + "tix", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));
                b_botix = new System.IO.BinaryWriter(t_botix = new System.IO.FileStream(guid + "bix", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));
                b_linked = new System.IO.BinaryWriter(t_linked = new System.IO.FileStream(guid + "lk", System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None));

                w_tlg.Write((byte)0x41);
                w_tlg.Write((ushort)0x7);
                w_tlg.Write(SectionTag);
                Section_Tracks_pos = w_tlg.BaseStream.Position;
                w_tlg.Write((long)0);

                w_tlg.Write(id.Part0);
                w_tlg.Write(id.Part1);
                w_tlg.Write(id.Part2);
                w_tlg.Write(id.Part3);

                w_tlg.Write(center.X);
                w_tlg.Write(center.Y);

                w_tlg.Write(extents.MinX);
                w_tlg.Write(extents.MaxX);
                w_tlg.Write(extents.MinY);
                w_tlg.Write(extents.MaxY);

                w_tlg.Write(transform.MXX);
                w_tlg.Write(transform.MXY);
                w_tlg.Write(transform.MYX);
                w_tlg.Write(transform.MYY);
                w_tlg.Write(transform.TX);
                w_tlg.Write(transform.TY);
                w_tlg.Write(transform.RX);
                w_tlg.Write(transform.RY);
            }

            static byte [] empty_tk = new byte[MIPIndexedEmulsionTrack.Size];
            static byte [] empty_ix = new byte[MIPIndexedEmulsionTrack.IndexSize];
            static byte [] empty_lk = new byte[MIPBaseTrack.Size];
            static byte [] empty_vw = new byte[View.Size];

            public int TopTracks { get { return toptks; } }

            public int TopViews { get { return topvws; } }

            public int BottomTracks { get { return bottks; } }

            public int BottomViews { get { return botvws; } }

            public int Linked { get { return linked; } }


            public void AddMIPEmulsionTrack(SySal.Tracking.MIPEmulsionTrackInfo info, int id, int viewid, SySal.Scanning.Plate.IO.OPERA.LinkedZone.TrackIndexEntry tie, bool istop)
            {
                System.IO.FileStream t_strtk, t_strix;
                System.IO.BinaryWriter b_strtk, b_strix;
                int strtks = 0;
                if (istop)
                {
                    t_strtk = t_toptk;
                    b_strtk = b_toptk;
                    t_strix = t_topix;
                    b_strix = b_topix;
                    strtks = toptks;
                }
                else
                {
                    t_strtk = t_bottk;
                    b_strtk = b_bottk;
                    t_strix = t_botix;
                    b_strix = b_botix;
                    strtks = bottks;
                }
                if (id < 0) throw new Exception("Only positive Ids are accepted. " + id + " is rejected.");
                if (id >= strtks)
                {
                    t_strtk.Seek(0, System.IO.SeekOrigin.End);
                    t_strix.Seek(0, System.IO.SeekOrigin.End);
                    while (strtks < id)                
                    {
                        b_strtk.Write(empty_tk);
                        b_strix.Write(empty_ix);
                        strtks++;
                    }
                }
                else
                {
                    t_strtk.Seek(((long)id) * MIPIndexedEmulsionTrack.Size, System.IO.SeekOrigin.Begin);
                    t_strix.Seek(((long)id) * MIPIndexedEmulsionTrack.IndexSize, System.IO.SeekOrigin.Begin);
                }
                long chkpos = t_strtk.Position;
                b_strtk.Write(info.Field);
                b_strtk.Write(info.AreaSum);
                b_strtk.Write((uint)info.Count);
                b_strtk.Write(info.Intercept.X);
                b_strtk.Write(info.Intercept.Y);
                b_strtk.Write(info.Intercept.Z);
                b_strtk.Write(info.Slope.X);
                b_strtk.Write(info.Slope.Y);
                b_strtk.Write(info.Slope.Z);
                b_strtk.Write(info.Sigma);
                b_strtk.Write(info.TopZ);
                b_strtk.Write(info.BottomZ);
                b_strtk.Write(viewid);
                if (t_strtk.Position - chkpos != MIPIndexedEmulsionTrack.Size) throw new Exception("B");
                chkpos = t_strix.Position;
                b_strix.Write(tie.Fragment);
                b_strix.Write(tie.View);
                b_strix.Write(tie.Track);
                if (t_strix.Position - chkpos != MIPIndexedEmulsionTrack.IndexSize) throw new Exception("C");
                strtks = Math.Max(id + 1, strtks);
                if (istop) toptks = strtks;
                else bottks = strtks;                
            }

            public void AddMIPBasetrack(SySal.Tracking.MIPEmulsionTrackInfo info, int id, int topid, int botid)
            {
                if (id < 0) throw new Exception("Only positive Ids are accepted. " + id + " is rejected.");
                if (id >= linked)
                {
                    t_linked.Seek(0, System.IO.SeekOrigin.End);
                    while (linked < id)
                    {
                        b_linked.Write(empty_lk);
                        linked++;
                    }
                }
                else
                {
                    t_linked.Seek(((long)id) * MIPBaseTrack.Size, System.IO.SeekOrigin.Begin);
                }
                long chkpos = t_linked.Position;
                b_linked.Write(info.AreaSum);
                b_linked.Write((uint)info.Count);
                b_linked.Write(info.Intercept.X);
                b_linked.Write(info.Intercept.Y);
                b_linked.Write(info.Intercept.Z);
                b_linked.Write(info.Slope.X);
                b_linked.Write(info.Slope.Y);
                b_linked.Write(info.Slope.Z);
                b_linked.Write(info.Sigma);
                b_linked.Write(topid);
                b_linked.Write(botid);
                if (t_linked.Position - chkpos != MIPBaseTrack.Size) throw new Exception("A");
                linked = Math.Max(id + 1, linked);
            }

            public void AddView(SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw, bool istop)
            {
                System.IO.FileStream t_strvw;
                System.IO.BinaryWriter b_strvw;
                int strvw = 0;
                if (istop)
                {
                    t_strvw = t_topvw;
                    b_strvw = b_topvw;
                    strvw = topvws;
                }
                else
                {
                    t_strvw = t_botvw;
                    b_strvw = b_botvw;
                    strvw = botvws;
                }
                int id = vw.Id;
                if (id < 0) throw new Exception("Only positive Ids are accepted. " + id + " is rejected.");
                if (id >= strvw)
                {
                    t_strvw.Seek(0, System.IO.SeekOrigin.End);
                    while (strvw < id)
                    {
                        b_strvw.Write(empty_vw);
                        strvw++;
                    }
                }
                else
                {
                    t_strvw.Seek(((long)id) * View.Size, System.IO.SeekOrigin.Begin);
                }
                long chkpos = t_strvw.Position;
                b_strvw.Write(id);
                b_strvw.Write(vw.Position.X);
                b_strvw.Write(vw.Position.Y);
                b_strvw.Write(vw.TopZ);
                b_strvw.Write(vw.BottomZ);
                if (t_strvw.Position - chkpos != View.Size) throw new Exception("D");
                strvw = Math.Max(id + 1, strvw);
                if (istop) topvws = strvw;
                else botvws = strvw;                
            }

            public void Cancel()
            {
                w_tlg.Close();
                try
                {
                    System.IO.File.Delete(m_FilePath);
                }
                catch (Exception) { }
                Dispose();
            }

            double m_toptz = 0.0, m_topbz = 0.0, m_bottz = 0.0, m_botbz = 0.0;
            bool zinfoset = false;

            public void SetZInfo(double toptz, double topbz, double bottz, double botbz)
            {
                m_topbz = topbz;
                m_toptz = toptz;
                m_botbz = botbz;
                m_bottz = bottz;
                zinfoset = true;
            }

            public void Complete()
            {
                if (zinfoset == false) throw new Exception("Z information not set - can't complete stream write.");
                w_tlg.Write(topvws);
                w_tlg.Write(botvws);
                w_tlg.Write(m_toptz);
                w_tlg.Write(m_topbz);
                w_tlg.Write(m_bottz);
                w_tlg.Write(m_botbz);
                FlushStream(t_topvw);
                FlushStream(t_botvw);
                w_tlg.Write(toptks);
                w_tlg.Write(bottks);
                w_tlg.Write(linked);
                FlushStream(t_toptk);
                FlushStream(t_bottk);
                FlushStream(t_linked);
                FlushStream(t_topix);
                FlushStream(t_botix);
                long nextpos = w_tlg.BaseStream.Position;
                w_tlg.Seek((int)Section_Tracks_pos, System.IO.SeekOrigin.Begin);
                w_tlg.Write(nextpos);
                w_tlg.Seek(0, System.IO.SeekOrigin.End);
                w_tlg.Flush();
                w_tlg.Close();
                Dispose();
            }

            static string[] sfx = new string[] { "ttk", "tvw", "tix", "btk", "bvw", "bix", "lk" };

            protected void FreeResources()
            {
                if (guid != null)
                {
                    try
                    {
                        t_toptk.Close();
                        t_topvw.Close();
                        t_topix.Close();
                        t_bottk.Close();
                        t_botvw.Close();
                        t_botix.Close();
                        t_linked.Close();
                    }
                    catch (Exception) {}
                    foreach (string s in sfx)
                        try
                        {
                            System.IO.File.Delete(guid + s);
                        }
                        catch (Exception x) { Console.WriteLine("DataStream - error freeing resources: " + x.ToString()); }
                    guid = null;
                }
            }

            protected void FlushStream(System.IO.FileStream fs)
            {
                fs.Seek(0, System.IO.SeekOrigin.Begin);
                const int bsize = 1024;
                int readb;
                byte [] b = new byte[bsize];
                while ((readb = fs.Read(b, 0, bsize)) > 0)
                    w_tlg.Write(b, 0, readb);
            }

            ~Writer()
            {
                FreeResources();
            }

            #region IDisposable Members

            public void Dispose()
            {
                FreeResources();
                GC.SuppressFinalize(this);
            }

            #endregion
        }
    }
}
