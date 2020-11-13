using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Serialization;
using SySal.Executables.TagPrimary;
using SySal.Processing.TagPrimary;
using SySal.Management;
using SySal.BasicTypes;
using SySal.Tracking;


namespace SySal.Processing.TagPrimary
{
    /// <summary>
    /// Configuration for TagPrimary.
    /// </summary>
    [Serializable]
    [XmlType("TagPrimary.Configuration")]
    public class Configuration : SySal.Management.Configuration, ICloneable//, ISerializable
    {
        /// <summary>
        /// Builds a configuration initialized with default parameters.
        /// </summary>
        public Configuration()
            : base("")
        {
            PositionToleranceSB = 100;
            PositionToleranceCS = 400;
            AngularToleranceSB = 0.03;
            AngularToleranceCS = 0.03;
            EledetPosTol = 40000;
            EledetAngTol = 0.07;
        }

        /// <summary>
        /// Builds a configuration initialized with default parameters, and with the specified name.
        /// </summary>
        /// <param name="name"></param>
        public Configuration(string name)
            : base(name)
        {
            PositionToleranceSB = 100;
            PositionToleranceCS = 400;
            AngularToleranceSB = 0.03;
            AngularToleranceCS = 0.03;
            EledetPosTol = 40000;
            EledetAngTol = 0.07;
        }

        /// <summary>
        /// Position tolerance used to find SB tracks inside the TS volume
        /// </summary>
        public double PositionToleranceSB;

        /// <summary>
        /// Position tolerance used to find CS tracks inside the TS volume
        /// </summary>
        public double PositionToleranceCS;

        /// <summary>
        /// Angular tolerance used to find SB tracks inside the TS volume
        /// </summary>
        public double AngularToleranceSB;

        // <summary>
        /// Angular tolerance used to find CS tracks inside the TS volume
        /// </summary>
        public double AngularToleranceCS;

        // <summary>
        /// Position tolerance used to connect Electronic detector prediction with emulsion tracks
        /// </summary>
        public double EledetPosTol;

        // <summary>
        /// Angular tolerance used to connect Electronic detector prediction with emulsion tracks
        /// </summary>
        public double EledetAngTol;
        // <summary>
        /// SySal.Tracking.MIPEmulsionTrackInfo used to store electronic detector muon fit
        /// </summary>

        public override object Clone()
        {
            Configuration C = new Configuration();
            C.EledetAngTol = this.EledetAngTol;
            C.AngularToleranceCS = this.AngularToleranceCS;
            C.AngularToleranceSB = this.AngularToleranceSB;
            C.EledetPosTol = this.EledetPosTol;
            C.PositionToleranceCS = this.PositionToleranceCS;
            C.PositionToleranceSB = this.PositionToleranceSB;
            C.Name = this.Name;
            return C;
        }
    }

    [Serializable]
    [XmlType("TagPrimary.PrimaryVertexTagger")]
    public class PrimaryVertexTagger : IManageable
    {

        public override string ToString()
        {
            return "Primary Vertex Tagging Algorithm";
        }
        [NonSerialized]

        private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        private SySal.TotalScan.Flexi.Track eledetMuonLinearFit;

        private SySal.TotalScan.Flexi.Track eledetMuonKalmanFit;

        private string eventType;

        private long eventID;
        // <summary>
        /// SySal.Tracking.MIPEmulsionTrackInfo used to store CS/SB tracks
        /// </summary>
        /// 

        private SySal.TotalScan.Flexi.Track[] CSdata;
        private SySal.TotalScan.Flexi.Track[] SBdata;

        private SySal.TotalScan.Flexi.Track[] CSdataInVolume;
        private SySal.TotalScan.Flexi.Track[] SBdataInVolume;

        //INPUT/OUTPUT TSR
        private SySal.TotalScan.Volume inputTSR;
        // public SySal.TotalScan.Volume outputTSR;


        // <summary>
        /// set to true if the primary is found
        /// </summary>
        private bool isPrimary;

        #region IManageable Members

        protected Configuration C = new Configuration();

        /// <summary>
        /// The configuration of the primary tagging algorithm.
        /// </summary>
        public SySal.Management.Configuration Config
        {
            get
            {
                return (SySal.Management.Configuration)(C.Clone());
            }
            set
            {
                C = (SySal.Processing.TagPrimary.Configuration)(value.Clone());
            }
        }

        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            EditConfigForm ec = new EditConfigForm();
            ec.C = (SySal.Processing.TagPrimary.Configuration)(c.Clone());
            if (ec.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                c = (SySal.Processing.TagPrimary.Configuration)(ec.C.Clone());
                return true;
            }
            return false;
        }

        public bool MonitorEnabled
        {
            get
            {
                return false;
            }
            set
            {
                return;
            }
        }


        /// <summary>
        /// List of connections. It is always empty for TagPrimary.
        /// </summary>
        public IConnectionList Connections
        {
            get { return EmptyConnectionList; }
        }

        internal class SingleProngVertex : SySal.TotalScan.Flexi.Vertex
        {
            public SingleProngVertex(SySal.BasicTypes.Vector w, double avgd, SySal.TotalScan.Flexi.DataSet ds, int id, SySal.TotalScan.Track tk)
                : base(ds, id)
            {
                m_X = w.X;
                m_Y = w.Y;
                m_Z = w.Z;
                m_DX = m_DY = 0.0;
                m_AverageDistance = 0.0;
                m_VertexCoordinatesUpdated = true;
                Tracks = new SySal.TotalScan.Track[1] { tk };
            }
        }


        /// <summary>
        /// Member field on which the Name property relies.
        /// </summary>
        [NonSerialized]
        protected string m_Name;
        /// <summary>
        /// The name of the momentum estimator.
        /// </summary>
        public string Name
        {
            get
            {
                return (string)(m_Name.Clone());
            }
            set
            {
                m_Name = (string)(value.Clone());
            }
        }

        #endregion


        public struct TagPrimaryResult
        {
            /// <summary>
            /// Number of scanback paths followed in brick.
            /// </summary>
            public int ScanBackPaths;
            /// <summary>
            /// Number of scanback tracks recostructed in TotalScan data.
            /// </summary>
            public int ScanBackPathsInVolume;
            /// <summary>
            /// Number of CS paths followed in brick.
            /// </summary>
            public int CSPaths;
            /// <summary>
            /// Number of CS tracks recostructed in TotalScan data.
            /// </summary>
            public int CSPathsInVolume;
            /// <summary>
            /// Number of muon candidates.
            /// </summary>
            public int MuonCandidates;
            /// <summary>
            /// Vertex ID.
            /// </summary>
            public int VertexId;
            /// <summary>
            /// Number of prong.
            /// </summary>
            public int Prongs;
            /// <summary>
            /// Is the primary vertex found? (true/false).
            /// </summary>
            public bool IsFound;
            /// <summary>
            /// eventType (CC or NC).
            /// </summary>
            public string EventType;
            /// <summary>
            /// eventID.
            /// </summary>
            public long EventId;
        }

        //read TSR 
        public void ReadTSR(string inputFileTSR)
        {
            
            SySal.TotalScan.NamedAttributeIndex.RegisterFactory();
            SySal.TotalScan.BaseTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBNamedAttributeIndex.RegisterFactory();

            SySal.TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
            ds.DataType = "TSR";
            ds.DataId = 0;
            System.IO.FileStream r = new System.IO.FileStream(inputFileTSR, System.IO.FileMode.Open, System.IO.FileAccess.Read);
            inputTSR = new SySal.TotalScan.Flexi.Volume();
            ((SySal.TotalScan.Flexi.Volume)(inputTSR)).ImportVolume(ds, new SySal.TotalScan.Volume(r));
            r.Close();
        }

        //save TSR using Save function of Flexi.Volume class
        public void WriteTSR(string outputFileTSR)
        {
            System.IO.FileStream ws = new System.IO.FileStream(outputFileTSR, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
            ((SySal.TotalScan.Flexi.Volume)(inputTSR)).Save(ws);
            ws.Flush();
            ws.Close();
            ws = null;
        }

        // set and get eventID (used to retrieve event info from DataBase. Maybe to be removed: not used)
        internal void setEventID(long eventId)
        {
            eventID = eventId;
        }
        internal long getEventID()
        {
            return eventID;
        }

        // get ScanBack paths from TSR file (if any) (dataset SBSF)
        private int getSBfromTSR()
        {
            System.Collections.ArrayList sblist = new System.Collections.ArrayList();
            int nTracks = inputTSR.Tracks.Length;
            for (int i = 0; i < nTracks; i++)
            {
                SySal.TotalScan.Flexi.Track tr = ((SySal.TotalScan.Flexi.Track)inputTSR.Tracks[i]);
                if (tr.DataSet.DataType == "SBSF")
                {
                    if (!checkPassing(tr))
                        sblist.Add(tr);
                }
            }
            SBdata = (SySal.TotalScan.Flexi.Track[])sblist.ToArray(typeof(SySal.TotalScan.Flexi.Track));
            return SBdata.Length;
        }

        // get CS paths from TSR file (dataset CS)
        private int getCSfromTSR()
        {
            System.Collections.ArrayList cslist = new System.Collections.ArrayList();
            int nTracks = inputTSR.Tracks.Length;
            for (int i = 0; i < nTracks; i++)
            {
                SySal.TotalScan.Flexi.Track tr = ((SySal.TotalScan.Flexi.Track)inputTSR.Tracks[i]);
                if (tr.DataSet.DataType == "CS") // exclude SB tracks passing the scanned volume
                {
                    cslist.Add(tr);
                }
            }
            CSdata = (SySal.TotalScan.Flexi.Track[])cslist.ToArray(typeof(SySal.TotalScan.Flexi.Track));
            return CSdata.Length;
        }

        // retrieve Event info from the DataBase (CC/NC Type and Kalman/Linear eledet info)
        private void retrieveEventInfoFromDB()
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                //SySal.TotalScan.Flexi.DataSet dsinfo = new SySal.TotalScan.Flexi.DataSet();
                //dsinfo.DataId = System.Convert.ToInt64(dstext[1]);
                //dsinfo.DataType = "ELEDET";
                System.Data.DataSet ds1 = new System.Data.DataSet();
                string sqlQuery = "select tb_predicted_events.id,  tb_predicted_events.type, tv_predtrack_brick_assoc.track," +
                    "tv_predtrack_brick_assoc.type, tb_predicted_tracks.posx,tb_predicted_tracks.posy,tb_predicted_tracks.slopex," +
                    "tb_predicted_tracks.slopey from tb_predicted_events, tv_predtrack_brick_assoc,tb_predicted_tracks where event = " + eventID.ToString() +
                    " and tb_predicted_events.id = tv_predtrack_brick_assoc.id_event and tv_predtrack_brick_assoc.track=tb_predicted_tracks.track and tv_predtrack_brick_assoc.id_event = tb_predicted_tracks.id_event";

                new SySal.OperaDb.OperaDbDataAdapter(sqlQuery, conn).Fill(ds1);
                if (ds1.Tables[0].Rows.Count <= 0) return;
                eventType = (string)(ds1.Tables[0].Rows[0][1]);
                System.Collections.ArrayList eleDetList = new System.Collections.ArrayList();
                TotalScan.Flexi.DataSet ds = new SySal.TotalScan.Flexi.DataSet();
                ds.DataType = "TT";
                ds.DataId = 1;

                if (eventType == "CC")
                {
                    for (int i = 0; i < ds1.Tables[0].Rows.Count; i++)
                    {
                        System.Data.DataRow dr = ds1.Tables[0].Rows[i];
                        if ((string)(dr[3]) == "KALMAN")
                        {
                            eledetMuonKalmanFit = new SySal.TotalScan.Flexi.Track(ds,inputTSR.Tracks.Length);
                            MIPEmulsionTrackInfo temp = new MIPEmulsionTrackInfo();
                            temp.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[4]) * 1000;
                            temp.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[5]) * 1000;
                            temp.Intercept.Z = 4800f;
                            temp.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[6]);
                            temp.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[7]);
                            temp.Slope.Z = 1f;
                            temp.TopZ = temp.Intercept.Z;
                            temp.BottomZ = temp.Intercept.Z;
                            eledetMuonKalmanFit.AddSegment(new SySal.TotalScan.Segment(temp, new SySal.TotalScan.NullIndex()));
                            eleDetList.Add(eledetMuonKalmanFit);

                        }
                        else if ((string)(dr[3]) == "LINEAR")
                        {
                            eledetMuonLinearFit = new SySal.TotalScan.Flexi.Track(ds, inputTSR.Tracks.Length + 1);
                            MIPEmulsionTrackInfo temp = new MIPEmulsionTrackInfo();
                            temp.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[4])*1000;
                            temp.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[5])*1000;
                            temp.Intercept.Z = 4800f;
                            temp.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[6]);
                            temp.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[7]);
                            temp.Slope.Z = 1f;
                            eledetMuonLinearFit.AddSegment(new SySal.TotalScan.Segment(temp, new SySal.TotalScan.NullIndex()));
                            eleDetList.Add(eledetMuonLinearFit);
                        }
                        else
                        {
                            Console.WriteLine("Electronic Detector Muon Fit NOT found!\n");
                        }
                        
                                                
                        //
                    }
                    //TotalScan.Flexi.Track a = new SySal.TotalScan.Flexi.Track(ds, 1);
                    //SySal.TotalScan.Flexi.Track[] trArr = (SySal.TotalScan.Flexi.Track[])eleDetList.ToArray(typeof(SySal.TotalScan.Flexi.Track));


//                    ((SySal.TotalScan.Flexi.Volume.TrackList)inputTSR.Tracks).Insert(trArr); //ad TT info in inputTSR

                }

            }
            catch (Exception x)
            {
                Console.WriteLine("DB import error:\n" + x.Message.ToString());
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
            }
        }
        
        // get Event info from TSR --> Looking For DataSet "TT", if exist ==> CC + get KalmanAndLinear, else NC
        private void getEventInfoFromTSR()
        {
            int nTracks = inputTSR.Tracks.Length;
            int counter = 0;
            for (int i = 0; i < nTracks; i++)
            {
                SySal.TotalScan.Flexi.Track tr = ((SySal.TotalScan.Flexi.Track)inputTSR.Tracks[i]);
                if (tr.DataSet.DataType == "TT") // exclude SB tracks passing the scanned volume
                {
                    SySal.TotalScan.Attribute[] a = tr.ListAttributes();
                    foreach (SySal.TotalScan.Attribute a1 in a)
                    {
                        if (a1.Index is SySal.TotalScan.NamedAttributeIndex && ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.StartsWith("TYPE_LINEAR"))
                        {
                            eledetMuonLinearFit = tr;
                            counter++;

                        }
                        else if (a1.Index is SySal.TotalScan.NamedAttributeIndex && ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.StartsWith("TYPE_KALMAN"))
                        {
                            eledetMuonKalmanFit = tr;
                            counter++;

                        }

                    }

                    
                }
            }
            if (counter != 0)
            {
                eventType = "CC";
            }
            else
                eventType = "NC";
        }

        // propagate the two tracks in the same plane and then check if the two tracks are the same
        private bool compareTracks(SySal.TotalScan.Track tr1, SySal.TotalScan.Track tr2, double resPos, double resAng)
        {
            bool same = false;
            double dPos;
            double dSlope;

            double xUpTr1 = (tr1.Upstream_SlopeX * (tr1.Upstream_Z - tr1.Upstream_PosZ) + tr1.Upstream_PosX);
            double yUpTr1 = (tr1.Upstream_SlopeY * (tr1.Upstream_Z - tr1.Upstream_PosZ) + tr1.Upstream_PosY);
            double zUpTr1 = tr1.Upstream_Z;

            double xUpTr2 = (tr2.Upstream_SlopeX * (tr2.Upstream_Z - tr2.Upstream_PosZ) + tr2.Upstream_PosX);
            double yUpTr2 = (tr2.Upstream_SlopeY * (tr2.Upstream_Z - tr2.Upstream_PosZ) + tr2.Upstream_PosY);
            double zUpTr2 = tr2.Upstream_Z;

            double dz = zUpTr1 - zUpTr2;



            if (dz != 0)
            {
                double posXtr1 = xUpTr1 - dz * 0.5f * tr1.Upstream_SlopeX;
                double posYtr1 = yUpTr1 - dz * 0.5f * tr1.Upstream_SlopeY;
                double posZtr1 = zUpTr1 - dz * 0.5f;
                double n1 = tr1.Upstream_SlopeX;
                double m1 = tr1.Upstream_SlopeY;
                double l1 = 1f;

                double posXtr2 = xUpTr2 + dz * 0.5f * tr2.Upstream_SlopeX;
                double posYtr2 = yUpTr2 + dz * 0.5f * tr2.Upstream_SlopeY;
                double posZtr2 = zUpTr2 + dz * 0.5f;
                double n2 = tr2.Upstream_SlopeX;
                double m2 = tr2.Upstream_SlopeY;
                double l2 = 1f;

                dPos = Math.Sqrt((posXtr1 - posXtr2) * (posXtr1 - posXtr2) + (posYtr1 - posYtr2) * (posYtr1 - posYtr2));
                dSlope = Math.Acos((m1 * m2 + n1 * n2 + l1 * l2) / Math.Sqrt((m1 * m1 + n1 * n1 + l1 * l1) * (m2 * m2 + n2 * n2 + l2 * l2)));
            }
            else
            {
                double n1 = tr1.Upstream_SlopeX;
                double m1 = tr1.Upstream_SlopeY;
                double l1 = 1f;

                double n2 = tr2.Upstream_SlopeX;
                double m2 = tr2.Upstream_SlopeY;
                double l2 = 1f;

                dPos = Math.Sqrt((xUpTr1 - xUpTr2) * (xUpTr1 - xUpTr2) + (yUpTr1 - yUpTr2) * (yUpTr1 - yUpTr2));
                dSlope = Math.Acos((m1 * m2 + n1 * n2 + l1 * l2) / Math.Sqrt((m1 * m1 + n1 * n1 + l1 * l1) * (m2 * m2 + n2 * n2 + l2 * l2)));
            }
            if (dPos < resPos && dSlope < resAng)
            {
                same = true;
            }

            return same;

        }

        // search muon inside SB paths (dataset SB)
        private int searchMuonInScanBackPaths()
        {
            int nCand = 0;

            for (int i = 0; i < SBdata.Length; i++)
            {
                SySal.TotalScan.Track tr = SBdata[i];
                if (compareTracks(tr, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(tr, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                {
                    tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                    nCand++;
                }
            }

            return nCand;
        }

        // search muon inside CS paths (dataset CS)
        private int searchMuonInCSPaths()
        {
            int nCand = 0;

            for (int i = 0; i < CSdata.Length; i++)
            {
                SySal.TotalScan.Track tr = CSdata[i];
                if (compareTracks(tr, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(tr, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                {
                    tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                    nCand++;
                }
            }

            return nCand;
        }

        // search in volume tracks (dataset TSR) looking for reconstructed ScanBack tracks (dataset SBSF)
        private int searchScanBackTracksInVolume()
        {
            int nTracks = inputTSR.Tracks.Length;
            int nSB = SBdata.Length;
            if (nSB == 0)
                return 0;
            System.Collections.ArrayList tklist = new System.Collections.ArrayList();

            for (int i = 0; i < nTracks; i++)
            {
                SySal.TotalScan.Flexi.Track trk = (SySal.TotalScan.Flexi.Track)(inputTSR.Tracks[i]);
                if (trk.DataSet.DataType == "TSR")
                {
                    for (int j = 0; j < nSB; j++)
                    {
                        if (compareTracks(trk, SBdata[j], C.PositionToleranceSB, C.AngularToleranceSB))
                            tklist.Add(trk);
                    }
                }
            }
            SBdataInVolume = (SySal.TotalScan.Flexi.Track[])tklist.ToArray(typeof(SySal.TotalScan.Flexi.Track));

            return SBdataInVolume.Length;

        }

        // search in volume tracks (dataset TSR) looking for reconstructed CS tracks (dataset CS)
        private int searchCSTracksInVolume()
        {
            int nTracks = inputTSR.Tracks.Length;
            int nCS = CSdata.Length;
            System.Collections.ArrayList tklist = new System.Collections.ArrayList();
            if (nCS == 0)
                return 0;
            for (int i = 0; i < nTracks; i++)
            {
                SySal.TotalScan.Flexi.Track trk = (SySal.TotalScan.Flexi.Track)(inputTSR.Tracks[i]);
                if (trk.DataSet.DataType == "TSR")
                {
                    for (int j = 0; j < nCS; j++)
                    {
                        if (compareTracks(trk, CSdata[j], C.PositionToleranceCS, C.AngularToleranceCS))
                        {
                            tklist.Add(trk);
                        }
                    }
                }
            }
            CSdataInVolume = (SySal.TotalScan.Flexi.Track[])tklist.ToArray(typeof(SySal.TotalScan.Flexi.Track));
            return CSdataInVolume.Length;

        }

        // search muon in SB track (if any flag it)
        private int searchMuonInScanBackTracks()
        {
            int nCand = 0;


            for (int i = 0; i < SBdataInVolume.Length; i++)
            {
                SySal.TotalScan.Track tr = SBdataInVolume[i];
                if (compareTracks(tr, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(tr, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                {
                    tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                    nCand++;
                }
            }

            return nCand;
        }

        // search muon in CS track (if any flag it)
        private int searchMuonInCSTracks()
        {
            int nCand = 0;


            for (int i = 0; i < CSdataInVolume.Length; i++)
            {
                SySal.TotalScan.Track tr = CSdataInVolume[i];
                if (compareTracks(tr, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(tr, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                {
                    tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                    nCand++;
                }
            }

            return nCand;
        }


        // search muon candidates in volume track and flag
        private int searchMuonInVolume()
        {
            int nCand = 0;
            int nTracks = inputTSR.Tracks.Length;
            for (int i = 0; i < nTracks; i++)
            {
                SySal.TotalScan.Flexi.Track trk = (SySal.TotalScan.Flexi.Track)(inputTSR.Tracks[i]);
                if (trk.DataSet.DataType == "TSR")
                {
                    if (compareTracks(trk, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(trk, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                    {
                        trk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                        nCand++;
                    }
                }
            }
            return nCand;
        }

        // check if a track is passing the volume
        private bool checkPassing(SySal.TotalScan.Track tr)
        {

            bool isPassing = false;
            if (tr != null)
            {
                int nLayers = inputTSR.Layers.Length;
                double zMin = 0;
                int upstreamLayerId = 0;
                // get upstreamZ (inputTSR.Extents doesn't work....)
                
                for (int i = nLayers - 1; i > 0; i--)
                {
                    if (inputTSR.Layers[i].Length != 0)
                    {
                        zMin = inputTSR.Layers[i].UpstreamZ;
                        upstreamLayerId = i;
                        break;
                    }
                }
                int nSeg = tr.Length;
                SySal.TotalScan.Segment segLast = tr[nSeg - 1];

                //check upstream:


                if (segLast.LayerOwner.UpstreamZ - zMin < 0)
                {
                    isPassing = true;

                }
            }
            return isPassing;

        }

        // get Upstream track among an array of tracks: if more tracks at same z return the longest one
        private SySal.TotalScan.Track getUpstreamTrack(SySal.TotalScan.Track[] trArr)
        {
            int index = 0;
            int nSeg = 0;
            double minZ = 0;
            System.Collections.ArrayList tklist = new System.Collections.ArrayList();

            for (int i = 0; i < trArr.Length; i++)
            {
                if (trArr[i].Upstream_Z <= minZ && trArr[i].Length > nSeg)
                {
                    nSeg = trArr[i].Length;
                    minZ = trArr[i].Upstream_Z;
                    index = i;
                }
            }
            return trArr[index];
        }

        // check if a vertex has a parent track: if there is any get it otherwise return a null pointer)
        private SySal.TotalScan.Track getParentTrack(SySal.TotalScan.Vertex vtx)
        {

            SySal.TotalScan.Track tr = null;
            if (vtx != null)
            {
                int nTracks = vtx.Length;
                for (int i = 0; i < nTracks; i++)
                {

                    if (vtx[i].Downstream_Z < vtx.Z)
                    {
                        tr = vtx[i];
                        return tr;
                    }

                }
            }
            return tr;
        }

        // Process a SySal.TotalScan.Volume looking for primary vertex
        public int ProcessData(SySal.TotalScan.Volume invol)
        {
            inputTSR = invol;
            SySal.Processing.TagPrimary.PrimaryVertexTagger.TagPrimaryResult outputInfo = SearchPrimaryVertex();

            if (!outputInfo.IsFound)
            {
                if (outputInfo.EventType != "CC" && outputInfo.EventType != "NC")
                    throw new Exception("Event Type not defined.");
                return -1;
            }
            return outputInfo.VertexId;
        }



        // Algorithm implementation ==> Work In Progress
        public TagPrimaryResult SearchPrimaryVertex()
        {
            isPrimary = false;   // not found at the beginning.....
            TagPrimaryResult myInfo = new TagPrimaryResult();


            //retrieveEventInfoFromDB(); // to be removed from there!!!
            getEventInfoFromTSR();
            int nScanBackPaths = getSBfromTSR();
            int nCSPaths = getCSfromTSR();
            if (nCSPaths == 0 && nScanBackPaths == 0)
            {
                myInfo.IsFound = false;
                return myInfo;

            }
            
            myInfo.CSPaths = nCSPaths;
            myInfo.ScanBackPaths = nScanBackPaths;
            myInfo.EventType = eventType;
            myInfo.EventId = eventID;

            if (eventType == "CC")
            {
                // retrieve SB paths.
                if (nScanBackPaths != 0)
                {
                    //scanback has been done

                    int nMuCandidates = searchMuonInScanBackPaths();
                    int nScanBackTracks = searchScanBackTracksInVolume();
                    myInfo.ScanBackPathsInVolume = nScanBackTracks;

                    if (nMuCandidates != 0) // muon has been followed in Scanback Mode!
                    {

                        if (searchMuonInScanBackTracks() > 0)
                        {
                            System.Collections.ArrayList trklist = new System.Collections.ArrayList();
                            for (int i = 0; i < nScanBackTracks; i++)
                            {
                                SySal.TotalScan.Track tr = SBdataInVolume[i];
                                SySal.TotalScan.Attribute[] a = tr.ListAttributes();
                                foreach (SySal.TotalScan.Attribute a1 in a)
                                {
                                    if (a1.Index is SySal.TotalScan.NamedAttributeIndex && ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.StartsWith("PARTICLE"))
                                        trklist.Add(tr);
                                }
                            }
                            SySal.TotalScan.Track[] trMuCand = (SySal.TotalScan.Track[])trklist.ToArray(typeof(SySal.TotalScan.Track));
                            myInfo.MuonCandidates = trMuCand.Length;
                            for (int i = 0; i < trMuCand.Length; i++)
                            {
                                SySal.TotalScan.Track tr = trMuCand[i];
                                SySal.TotalScan.Vertex vtx = tr.Upstream_Vertex;


                                while (!checkPassing(tr) && tr != null)
                                {
                                    if (vtx != null)
                                    {
                                        tr = getParentTrack(vtx);
                                        if (tr == null)
                                        {
                                            //vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PrimaryTagged"), 1f);
                                            vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                            vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                            isPrimary = true;
                                            //myInfo.isFound = isPrimary;
                                            myInfo.VertexId = vtx.Id;
                                            myInfo.Prongs = vtx.Length;
                                            for (int j = 0; j < vtx.Length; j++)
                                            {
                                                vtx[j].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                            }
                                        }
                                        else
                                            vtx = tr.Upstream_Vertex;
                                    }
                                    else
                                    {
                                        //QE interaction  --> Add New Vertex

                                        SySal.BasicTypes.Vector w = new SySal.BasicTypes.Vector();
                                        w.X = tr.Upstream_PosX + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeX;
                                        w.Y = tr.Upstream_PosY + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeY;
                                        w.Z = tr.Upstream_Z - 500f;

                                        SySal.TotalScan.Flexi.Vertex newvtx = new SingleProngVertex(w, 0.0, ((SySal.TotalScan.Flexi.Track)tr).DataSet, inputTSR.Vertices.Length, tr);
                                        tr.SetUpstreamVertex(newvtx);
                                        tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                        newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                        newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                        ((SySal.TotalScan.Flexi.Volume.VertexList)inputTSR.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { newvtx });
                                        isPrimary = true;
                                        //myInfo.isFound = isPrimary;
                                        myInfo.Prongs = 1;
                                        myInfo.VertexId = newvtx.Id;
                                        tr = null;

                                    }
                                }
                            }
                        }
                        
                    }
                    else  // no muon among sb paths
                    {
                        if (SBdataInVolume.Length != 0)
                        {
                            SySal.TotalScan.Track tr = getUpstreamTrack(SBdataInVolume);
                            SySal.TotalScan.Vertex vtx = tr.Upstream_Vertex;

                            while (!checkPassing(tr) && tr != null)
                            {
                                if (vtx != null)
                                {
                                    tr = getParentTrack(vtx);

                                    if (tr == null)
                                    {
                                        //check if a track is compatible with eledetMuon
                                        
                                        for (int i = 0; i < vtx.Length; i++)
                                        {
                                            SySal.TotalScan.Track trTmp = vtx[i];
                                            if (compareTracks(trTmp, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(trTmp, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                                            {
                                                trTmp.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                                                vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                                vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                                isPrimary = true;
                                                myInfo.MuonCandidates = 1;
                                                myInfo.VertexId = vtx.Id;
                                                myInfo.Prongs = vtx.Length;
                                                for (int j = 0; j < vtx.Length; j++)
                                                {
                                                    vtx[j].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                                }
                                                break;
                                            }
                                        }
                                        //myInfo.isFound = isPrimary;
                                        
                                    }
                                    else
                                        vtx = tr.Upstream_Vertex;

                                }
                                else
                                {
                                    tr = null; // exit loop (primary NOT found)
                                    myInfo.IsFound = false;
                                }
                            }

                        }
                    }
                }
                else // NO SB --> Check From CS-data
                {
                    
                    int nMuCandidates = searchMuonInCSPaths();
                    int nCSTracks = searchCSTracksInVolume();
                    myInfo.CSPathsInVolume = nCSTracks;

                    if (nMuCandidates != 0) // muon candidate in CS paths
                    {
                        if (searchMuonInCSTracks() > 0)
                        {
                            System.Collections.ArrayList trklist = new System.Collections.ArrayList();
                            for (int i = 0; i < nCSTracks; i++)
                            {
                                SySal.TotalScan.Track tr = CSdataInVolume[i];
                                SySal.TotalScan.Attribute[] a = tr.ListAttributes();
                                foreach (SySal.TotalScan.Attribute a1 in a)
                                {
                                    if (a1.Index is SySal.TotalScan.NamedAttributeIndex && ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.StartsWith("PARTICLE"))
                                        trklist.Add(tr);
                                }
                            }
                            SySal.TotalScan.Track[] trMuCand = (SySal.TotalScan.Track[])trklist.ToArray(typeof(SySal.TotalScan.Track));
                            myInfo.MuonCandidates = trMuCand.Length;
                            for (int i = 0; i < trMuCand.Length; i++)
                            {
                                SySal.TotalScan.Track tr = trMuCand[i];
                                SySal.TotalScan.Vertex vtx = tr.Upstream_Vertex;


                                while (!checkPassing(tr) && tr != null)
                                {
                                    if (vtx != null)
                                    {
                                        tr = getParentTrack(vtx);
                                        if (tr == null)
                                        {
                                            vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                            vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                            
                                            myInfo.VertexId = vtx.Id;
                                            myInfo.Prongs = vtx.Length;
                                            isPrimary = true;
                                            //myInfo.isFound = isPrimary;
                                            for (int j = 0; j < vtx.Length; j++ )
                                                vtx[j].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                                
                                        }
                                        else
                                            vtx = tr.Upstream_Vertex;
                                    }
                                    else
                                    {
                                        //QE interaction  --> Add New Vertex

                                        SySal.BasicTypes.Vector w = new SySal.BasicTypes.Vector();
                                        w.X = tr.Upstream_PosX + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeX;
                                        w.Y = tr.Upstream_PosY + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeY;
                                        w.Z = tr.Upstream_Z - 500f;

                                        SySal.TotalScan.Flexi.Vertex newvtx = new SingleProngVertex(w, 0.0, ((SySal.TotalScan.Flexi.Track)tr).DataSet, inputTSR.Vertices.Length, tr);
                                        tr.SetUpstreamVertex(newvtx);
                                        tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                        newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                        newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                        ((SySal.TotalScan.Flexi.Volume.VertexList)inputTSR.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { newvtx });

                                        
                                        myInfo.VertexId = newvtx.Id;
                                        myInfo.Prongs = 1;

                                        isPrimary = true;
                                        //myInfo.isFound = isPrimary;
                                        tr = null;
                                    }
                                }
                            }
                        }
                    }
                    else // muon candidate NOT in CS paths
                    {
                        if (CSdataInVolume.Length != 0)
                        {
                            SySal.TotalScan.Track tr = getUpstreamTrack(CSdataInVolume);
                            SySal.TotalScan.Vertex vtx = tr.Upstream_Vertex;

                            while (!checkPassing(tr) && tr != null)
                            {
                                if (vtx != null)
                                {
                                    tr = getParentTrack(vtx);

                                    if (tr == null)
                                    {
                                        //check if a track is compatible with eledetMuon
                                        for (int i = 0; i < vtx.Length; i++)
                                        {
                                            SySal.TotalScan.Track trTmp = vtx[i];
                                            if (compareTracks(trTmp, eledetMuonKalmanFit, C.EledetPosTol, C.EledetAngTol) || compareTracks(trTmp, eledetMuonLinearFit, C.EledetPosTol, C.EledetAngTol))
                                            {
                                                trTmp.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PARTICLE"), 13f);
                                                vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                                vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);

                                                
                                                myInfo.VertexId = vtx.Id;
                                                myInfo.Prongs = vtx.Length;
                                                myInfo.MuonCandidates = 1;

                                                for (int j = 0; j < vtx.Length; j++)
                                                    vtx[j].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);

                                                isPrimary = true;
                                                //myInfo.isFound = isPrimary;
                                            }
                                        }
                                        


                                    }
                                    else
                                        vtx = tr.Upstream_Vertex;

                                }
                                else
                                {
                                    tr = null; // exit loop (primary NOT found)
                                    myInfo.IsFound = false;
                                }
                            }

                        }
                    }
                    
                       
                }
            }
            else if (eventType == "NC")
            {
                if (nScanBackPaths != 0)
                {
                    //scanback performed
                    int nTracks = searchScanBackTracksInVolume();
                    myInfo.ScanBackPathsInVolume = nTracks;

                    if (nTracks != 0)
                    {
                        SySal.TotalScan.Track tr = getUpstreamTrack(SBdataInVolume);


                        SySal.TotalScan.Vertex vtx = tr.Upstream_Vertex;


                        while (!checkPassing(tr) && tr != null)
                        {
                            if (vtx != null)
                            {
                                tr = getParentTrack(vtx);
                                if (tr == null)
                                {
                                    vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                    vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                    isPrimary = true;
                                    //myInfo.isFound = isPrimary;
                                    myInfo.VertexId = vtx.Id;
                                    myInfo.Prongs = vtx.Length;
                                    myInfo.MuonCandidates = 0;
                                    int nTracksInVtx = vtx.Length;
                                    for (int i = 0; i < nTracksInVtx; i++)
                                    {
                                        vtx[i].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                    }
                                }
                                else
                                    vtx = tr.Upstream_Vertex;
                            }
                            else
                            {

                                SySal.BasicTypes.Vector w = new SySal.BasicTypes.Vector();
                                w.X = tr.Upstream_PosX + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeX;
                                w.Y = tr.Upstream_PosY + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeY;
                                w.Z = tr.Upstream_Z - 500f;
                                SySal.TotalScan.Flexi.Vertex newvtx = new SingleProngVertex(w, 0.0, ((SySal.TotalScan.Flexi.Track)tr).DataSet, inputTSR.Vertices.Length, tr);
                                tr.SetUpstreamVertex(newvtx);
                                tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                ((SySal.TotalScan.Flexi.Volume.VertexList)inputTSR.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { newvtx });
                                isPrimary = true;
                                myInfo.MuonCandidates = 0;
                                myInfo.VertexId = newvtx.Id;
                                myInfo.Prongs = 1;
                                tr = null;

                            }
                        }
                    }
                    /*
                    else
                    {
                        myInfo.isFound = false;
                        isPrimary = false;
                    }
                    */
                }
                else
                {
                    // direct vertexing
                    int nCSTracks = searchCSTracksInVolume();
                    myInfo.CSPathsInVolume = nCSTracks;

                    if (nCSTracks != 0)
                    {
                        // TRACCE DEL CS TROVATE NEL BRICK --> NC Event
                        SySal.TotalScan.Track tr = getUpstreamTrack(CSdataInVolume);
                        SySal.TotalScan.Vertex vtx = tr.Upstream_Vertex;


                        while (!checkPassing(tr) && tr != null)
                        {
                            if (vtx != null)
                            {
                                tr = getParentTrack(vtx);
                                if (tr == null)
                                {
                                    vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                    vtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                    myInfo.VertexId = vtx.Id;
                                    myInfo.Prongs = vtx.Length;
                                    isPrimary = true;
                                    //myInfo.isFound = isPrimary;
                                    int nTracksInVtx = vtx.Length;
                                    for (int i = 0; i < nTracksInVtx; i++)
                                    {
                                        vtx[i].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                    }
                                }
                                else
                                    vtx = tr.Upstream_Vertex;
                            }
                            else
                            {

                                SySal.BasicTypes.Vector w = new SySal.BasicTypes.Vector();
                                w.X = tr.Upstream_PosX + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeX;
                                w.Y = tr.Upstream_PosY + (tr.Upstream_Z - tr.Upstream_PosZ - 500f) * tr.Upstream_SlopeY;
                                w.Z = tr.Upstream_Z - 500f;
                                SySal.TotalScan.Flexi.Vertex newvtx = new SingleProngVertex(w, 0.0, ((SySal.TotalScan.Flexi.Track)tr).DataSet, inputTSR.Vertices.Length, tr);
                                tr.SetUpstreamVertex(newvtx);
                                tr.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("PRIMARY"), 1f);
                                newvtx.SetAttribute(new SySal.TotalScan.NamedAttributeIndex("EVENT"), eventID);
                                ((SySal.TotalScan.Flexi.Volume.VertexList)inputTSR.Vertices).Insert(new SySal.TotalScan.Flexi.Vertex[1] { newvtx });
                                isPrimary = true;
                                //myInfo.isFound = isPrimary;
                                myInfo.VertexId = newvtx.Id;
                                myInfo.Prongs = 1;
                                tr = null;

                            }
                        }
                    }
                    /*
                    else
                    {
                        myInfo.isFound = false;
                        isPrimary = false;
                    }
                    */
                
                }
            }
            else
                Console.WriteLine("eventType NOT defined: impossible to find primary!");
            
            myInfo.IsFound = isPrimary; // check if ok.
            return myInfo;
        }
    }
}