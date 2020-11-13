using System;
using SySal.Tracking;
using SySal.Scanning.Plate.IO.OPERA;
using NumericalTools;

namespace SySal.Executables.TLGSelDS
{
    class MIPIndexedEmulsionTrack : SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack
    {
        public MIPIndexedEmulsionTrack(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack t, int id)
        {
            m_Info = TLGSelDS.MIPIndexedEmulsionTrack.AccessInfo(t);
            m_Grains = TLGSelDS.MIPIndexedEmulsionTrack.AccessGrains(t);
            m_Id = id;
            m_OriginalRawData = t.OriginalRawData;
        }

        public void SetView(View v)
        {
            m_View = v;
        }
    }

    class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
    {
        public MIPBaseTrack(SySal.Scanning.MIPBaseTrack t, int id)
        {
            m_Info = TLGSelDS.MIPBaseTrack.AccessInfo(t);
            m_Id = id;
            m_Top = t.Top;
            m_Bottom = t.Bottom;
            //View v;
            //m_Top = new TLGSelDS.MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)t.Top, t.Top.Id);
            //m_Bottom = new TLGSelDS.MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)t.Bottom, t.Bottom.Id);
        }

        public MIPBaseTrack(SySal.Scanning.MIPBaseTrack t, int id, TLGSelDS.Side top, TLGSelDS.Side bottom)
        {
            m_Info = TLGSelDS.MIPBaseTrack.AccessInfo(t);
            m_Id = id;
            View v;
            m_Top = new TLGSelDS.MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)t.Top, id);
            (v = (View)top.View(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)t.Top).View.Id)).AddTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)m_Top);
            ((TLGSelDS.MIPIndexedEmulsionTrack)m_Top).SetView(v);
            m_Bottom = new TLGSelDS.MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)t.Bottom, id);
            (v = (View)bottom.View(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)t.Bottom).View.Id)).AddTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)m_Bottom);
            ((TLGSelDS.MIPIndexedEmulsionTrack)m_Bottom).SetView(v);
        }
    }

    class View : SySal.Scanning.Plate.IO.OPERA.LinkedZone.View
    {
        System.Collections.ArrayList m_TkList = new System.Collections.ArrayList();

        public View(SySal.Scanning.Plate.IO.OPERA.LinkedZone.View v)
        {
            this.m_Id = v.Id;
            this.m_TopZ = v.TopZ;
            this.m_BottomZ = v.BottomZ;
            this.m_Position = v.Position;
        }

        public void SetSide(SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side s)
        {
            this.m_Side = s;
        }

        public void AddTrack(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack tk)
        {
            m_TkList.Add(tk);
        }

        public void FreezeTracks()
        {
            m_Tracks = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[])(m_TkList.ToArray(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)));
            m_TkList.Clear();
        }
    }

    class Side : SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side
    {
        public Side(SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side s)
        {
            m_TopZ = s.TopZ;
            m_BottomZ = s.BottomZ;
            m_Views = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.View[s.ViewCount];
            int i;
            for (i = 0; i < m_Views.Length; i++)
            {
                m_Views[i] = new View(s.View(i));
                ((View)m_Views[i]).SetSide(this);
            }
        }

        public void SetTracks(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack[] tks)
        {
            m_Tracks = tks;
        }

        public void FreezeViewTracks()
        {
            int i;
            for (i = 0; i < m_Views.Length; i++)
                ((View)m_Views[i]).FreezeTracks();
        }
    }

    class LinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
    {
        public static void Select(string outpath, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, CStyleParsedFunction f, bool cleanmicrotracks, int[] includelist, int[] excludelist, out int[] basetkremaplist, out int[] toptkremaplist, out int[] bottomtkremaplist)
        {
            System.IO.StreamWriter biwr = null;
            SySal.DataStreams.OPERALinkedZone.Writer wr = null;
            if (outpath.ToLower().EndsWith(".bi"))
            {
                biwr = new System.IO.StreamWriter(outpath);
                biwr.WriteLine("Index");
            }
            else
            {
                wr = new SySal.DataStreams.OPERALinkedZone.Writer(outpath, lz.Id, lz.Extents, lz.Center, lz.Transform);
                wr.SetZInfo(lz.Top.TopZ, lz.Top.BottomZ, lz.Bottom.TopZ, lz.Bottom.BottomZ);
            }
            int i, p;

            System.Collections.ArrayList basetkremap = new System.Collections.ArrayList();
            System.Collections.ArrayList toptkremap = new System.Collections.ArrayList();
            System.Collections.ArrayList bottomtkremap = new System.Collections.ArrayList();

            bool[] shouldinclude = new bool[lz.Length];
            if (includelist == null)
                for (i = 0; i < shouldinclude.Length; shouldinclude[i++] = true) ;
            else
                foreach (int ii in includelist)
                    shouldinclude[ii] = true;
            if (excludelist != null)
                foreach (int ii in excludelist)
                    shouldinclude[ii] = false;

            string[] pars = f.ParameterList;

            int selected = 0;

            for (i = 0; i < lz.Length; i++)
            {
                if (shouldinclude[i])
                {
                    SySal.Scanning.MIPBaseTrack t = lz[i];
                    for (p = 0; p < pars.Length; p++)
                    {
                        switch (pars[p].ToUpper())
                        {
                            case "A": f[p] = (double)t.Info.AreaSum; break;
                            case "TA": f[p] = (double)t.Top.Info.AreaSum; break;
                            case "BA": f[p] = (double)t.Bottom.Info.AreaSum; break;

                            case "N": f[p] = (double)t.Info.Count; break;
                            case "TN": f[p] = (double)t.Top.Info.Count; break;
                            case "BN": f[p] = (double)t.Bottom.Info.Count; break;

                            case "PX": f[p] = (double)t.Info.Intercept.X; break;
                            case "TPX": f[p] = (double)t.Top.Info.Intercept.X; break;
                            case "BPX": f[p] = (double)t.Bottom.Info.Intercept.X; break;

                            case "PY": f[p] = (double)t.Info.Intercept.Y; break;
                            case "TPY": f[p] = (double)t.Top.Info.Intercept.Y; break;
                            case "BPY": f[p] = (double)t.Bottom.Info.Intercept.Y; break;

                            case "PZ": f[p] = (double)t.Info.Intercept.Z; break;
                            case "TPZ": f[p] = (double)t.Top.Info.Intercept.Z; break;
                            case "BPZ": f[p] = (double)t.Bottom.Info.Intercept.Z; break;

                            case "SX": f[p] = (double)t.Info.Slope.X; break;
                            case "TSX": f[p] = (double)t.Top.Info.Slope.X; break;
                            case "BSX": f[p] = (double)t.Bottom.Info.Slope.X; break;

                            case "SY": f[p] = (double)t.Info.Slope.Y; break;
                            case "TSY": f[p] = (double)t.Top.Info.Slope.Y; break;
                            case "BSY": f[p] = (double)t.Bottom.Info.Slope.Y; break;

                            case "S": f[p] = (double)t.Info.Sigma; break;
                            case "TS": f[p] = (double)t.Top.Info.Sigma; break;
                            case "BS": f[p] = (double)t.Bottom.Info.Sigma; break;

                            case "TF": f[p] = (double)((MIPIndexedEmulsionTrack)t.Top).OriginalRawData.Fragment; break;
                            case "TV": f[p] = (double)((MIPIndexedEmulsionTrack)t.Top).OriginalRawData.View; break;
                            case "TID": f[p] = (double)((MIPIndexedEmulsionTrack)t.Top).OriginalRawData.Track; break;

                            case "BF": f[p] = (double)((MIPIndexedEmulsionTrack)t.Bottom).OriginalRawData.Fragment; break;
                            case "BV": f[p] = (double)((MIPIndexedEmulsionTrack)t.Bottom).OriginalRawData.View; break;
                            case "BID": f[p] = (double)((MIPIndexedEmulsionTrack)t.Bottom).OriginalRawData.Track; break;

                            default: throw new Exception("Unknown parameter " + pars[p]);
                        }
                    }
                    if (f.Evaluate() > 0.0)
                    {
                        if (biwr != null) biwr.WriteLine(i);
                        else
                        {
                            basetkremap.Add(i);
                            if (cleanmicrotracks)
                            {
                                wr.AddMIPEmulsionTrack(lz[i].Top.Info, selected, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Top).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Top).OriginalRawData, true);
                                toptkremap.Add(lz[i].Top.Id);
                                wr.AddMIPEmulsionTrack(lz[i].Bottom.Info, selected, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Bottom).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Bottom).OriginalRawData, false);
                                bottomtkremap.Add(lz[i].Bottom.Id);
                                wr.AddMIPBasetrack(lz[i].Info, selected, selected, selected);
                                selected++;
                            }
                            else
                            {
                                int ti, bi;
                                wr.AddMIPEmulsionTrack(lz[i].Top.Info, ti = lz[i].Top.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Top).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Top).OriginalRawData, true);
                                wr.AddMIPEmulsionTrack(lz[i].Bottom.Info, bi = lz[i].Bottom.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Bottom).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Bottom).OriginalRawData, false);
                                wr.AddMIPBasetrack(lz[i].Info, selected, ti, bi);
                                selected++;
                            }
                        }
                    }
                }
            }
            if (wr != null)
            {
                for (i = 0; i < ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Top).ViewCount; i++)
                    wr.AddView(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Top).View(i), true);
                for (i = 0; i < ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Bottom).ViewCount; i++)
                    wr.AddView(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Bottom).View(i), false);
            }
            basetkremaplist = (int[])basetkremap.ToArray(typeof(int));
            if (cleanmicrotracks)
            {
                toptkremaplist = (int[])toptkremap.ToArray(typeof(int));
                bottomtkremaplist = (int[])bottomtkremap.ToArray(typeof(int));
            }
            else
            {
                toptkremaplist = bottomtkremaplist = null;
            }
            if (wr != null) wr.Complete();
            if (biwr != null)
            {
                biwr.Flush();
                biwr.Close();
            }
        }

    }

    /// <summary>
    /// TLGSelDS - Command line tool to select base-tracks in a LinkedZone.
    /// </summary>
    /// <remarks>
    /// <para>Base-tracks are selected on the basis of inclusion/exclusion lists and a user-defined selection function.
    /// The known parameters to set up this selection function are shown in the following table:
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
    /// <item><term>T,BF</term><description>Fragment index of the top/bottom microtrack.</description></item>
    /// <item><term>T,BV</term><description>View index of the top/bottom microtrack.</description></item>
    /// <item><term>T,BID</term><description>Number of the top/bottom microtrack in its own view.</description></item>
    /// </list>
    /// If the <c>/micro</c> option is specified, microtracks not associated with surviving base-tracks are deleted.
    /// </para>
    /// <para>
    /// If no inclusion list is specified, all tracks that are not explicitly excluded by the selection function or one exclusion list are selected. 
    /// If one or more inclusion lists are specified, only explicitly included tracks are eligible for selection, unless they are explicitly excluded by an exclusion list or the selection function.
    /// </para>
    /// <para>
    /// Inclusion and exclusion lists are specified using files with ASCII n-tuples. Each ASCII file can be headerless or have a header row with the names of the fields. In the former case, 
    /// the <i>field id</i> is specified as a number; in the second case, it is a case-insensitive string that must match one of the column names in the file header.
    /// </para>
    /// <example>
    /// <c>TLGSelDS.exe myfile.tlg myselfile.tlg "sqrt((sx-0.055)^2+(sy-0.003)^2) &lt; 0.03 /micro /i incl.txt 3 /x erase1.txt PID /x erase2.txt PID</c>
    /// <para>This would erase basetracks and microtracks from file <c>myfile.tlg</c>, including only the tracks specified in the 4th column of <c>incl.txt</c>, excluding all tracks that do not satisfy the selection or whose Id is found in the <c>PID</c> column of <c>erase1.txt</c> or <c>erase2.txt</c>.</para>
    /// </example>
    /// <example>
    /// <c>TLGSelDS.exe myfile.tlg myselfile.bi "sqrt((sx-0.055)^2+(sy-0.003)^2) &lt; 0.03 /i incl.txt 3 /x erase1.txt PID /x erase2.txt PID</c>
    /// <para>The same as the above example, but since the output extension is ".bi", an ASCII file with track indices will be generated instead of a complete TLG file.</para>
    /// </example>
    /// <para>TLGSelDS recognizes and handles the following TLG sections (only for MultiSection TLG files):
    /// <list type="bullet">
    /// <item><term><see cref="SySal.Scanning.Plate.IO.Opera.LinkedZone.BaseTrackIndex"/>: if the source TLG contains a list of base track indices, the link is maintained after the selection (i.e., the list in the output TLG is consistent with the original file, no double-indexing is needed). 
    /// If no BaseTrackIndex list is found, TLGSelDS appends automatically a section containing the indices of the selected tracks in the original file.</term></item>
    /// <item><term><see cref="SySal.Scanning.Plate.IO.Opera.LinkedZone.BaseTrackIgnoreAlignment"/></term>: if the source TLG contains a list of tracks to be ignored in alignment, TLGSelDS updates their indices in the output file, so that the selected tracks are correctly referred to.</item>
    /// <item><term><see cref="SySal.OperaDb.Scanning.DBMIPMicroTrackIndex"/>: if the source TLG contains a list of microtrack DB indices, and the <c>/micro</c> switch is enabled, the list of microtracks indices is updated to reflect the selection of microtracks.</term></item>
    /// </list>
    /// </para>
    /// </remarks>
    public class Exe
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main(string[] args)
        {
            int[] includelist = null;
            int[] excludelist = null;

            bool CleanMicroTracks = false;
#if (!(DEBUG))
            try
            {
#endif
                try
                {
                    System.Collections.ArrayList idl = new System.Collections.ArrayList();
                    int paramscan = 3;
                    int idpos;
                    int headerpos;
                    int linepos;
                    int idscan;
                    string header;
                    string line;
                    if (args.Length >= 4 && String.Compare(args[3], "/micro", true) == 0)
                    {
                        CleanMicroTracks = true;
                        paramscan = 4;
                    }
                    if (args.Length < 3) throw new Exception("Too few arguments.");
                    System.Text.RegularExpressions.Regex rgx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)");
                    System.Text.RegularExpressions.Match rgm;
                    while (paramscan < args.Length)
                    {
                        if (String.Compare(args[paramscan], "/i", true) == 0 || String.Compare(args[paramscan], "/x", true) == 0)
                        {
                            idl.Clear();
                            try
                            {
                                idpos = (int)Convert.ToUInt32(args[paramscan + 2]);
                            }
                            catch (Exception)
                            {
                                idpos = -1;
                            }
                            System.IO.StreamReader r = new System.IO.StreamReader(args[paramscan + 1]);
                            if (idpos < 0)
                            {
                                header = r.ReadLine();
                                headerpos = 0;
                                idpos = -1;
                                do
                                {
                                    rgm = rgx.Match(header, headerpos);
                                    if (rgm.Success)
                                        if (String.Compare(rgm.Groups[1].Value, args[paramscan + 2], true) == 0)
                                        {
                                            idpos = -idpos - 1;
                                            break;
                                        }
                                    idpos--;
                                    headerpos += rgm.Length;
                                }
                                while (rgm.Success && headerpos < header.Length);
                                if (idpos < 0) throw new Exception("Can't find field \"" + args[paramscan + 2] + "\".");
                            }
                            while ((line = r.ReadLine()) != null)
                            {
                                line = line.Trim();
                                if (line.Length == 0) continue;
                                linepos = 0;
                                rgm = null;
                                for (idscan = 0; idscan <= idpos; idscan++)
                                {
                                    rgm = rgx.Match(line, linepos);
                                    linepos += rgm.Length;
                                }
                                idl.Add(Convert.ToInt32(rgm.Groups[1].Value));
                            }
                            r.Close();
                            if (String.Compare(args[paramscan], "/i", true) == 0)
                            {
                                if (includelist == null) includelist = (int[])idl.ToArray(typeof(int));
                                else
                                {
                                    idl.AddRange(includelist);
                                    includelist = (int[])idl.ToArray(typeof(int));
                                }
                            }
                            else
                            {
                                if (excludelist == null) excludelist = (int[])idl.ToArray(typeof(int));
                                else
                                {
                                    idl.AddRange(excludelist);
                                    excludelist = (int[])idl.ToArray(typeof(int));
                                }
                            }
                            paramscan += 3;
                        }
                        else
                        {
                            throw new Exception("Unknown switch or syntax");
                        }
                    }
                }
                catch (Exception)
                {
                    Console.WriteLine("usage: TLGSelDS <input TLG file> <output file> <selection expression> [/micro] {/i <ASCII n-tuple file> <field id>} {/x <ASCII n-tuple file> <field id>}");
                    Console.WriteLine("The output can be a TLG file or an ASCII .bi file (indices of selected base tracks).");
                    Console.WriteLine("if /micro is specified, unconnected microtracks are deleted too (this has no meaning for .bi output).");
                    Console.WriteLine("valid parameters:");
                    Console.WriteLine("PX -> X Position\nPY -> Y Position\nSX -> X Slope\nSY -> Y Slope\nS -> Sigma\nN -> Points\nA -> Area Sum");
                    Console.WriteLine("TPX -> Top X Position\nTPY -> Top Y Position\nTSX -> Top X Slope\nTSY -> Top Y Slope\nTS -> Top Sigma\nTN -> Top Points\nTA -> Top Area Sum");
                    Console.WriteLine("BPX -> Bottom X Position\nBPY -> Bottom Y Position\nBSX -> Bottom X Slope\nBSY -> Bottom Y Slope\nBS -> Bottom Sigma\nBN -> Bottom Points\nBA -> Bottom Area Sum");
                    Console.WriteLine("TF -> Top Fragment Index\nTV -> Top View Index\nTID -> Top Track Id");
                    Console.WriteLine("BF -> Bottom Fragment Index\nBV -> Bottom View Index\nBID -> Bottom Track Id");
                    Console.WriteLine("/i specifies an inclusion list, using an ASCII n-tuple file. More than one inclusion list can be specified.");
                    Console.WriteLine("/x specifies an exclusion list, using an ASCII n-tuple file. More than one exclusion list can be specified.");
                    Console.WriteLine("For both /i and /x, <field id> specifies the n-tuple field to read the Id from. If <field id> is a number, it is the zero-based index of the column containing the Id; if it is a string, it is the case-insensitive name of the column, in the ASCII header, that contains the Id.");
                    return;
                }

                /*
                                CStyleParsedFunction f = new CStyleParsedFunction(args[2]);
                                //TLGSelDS.LinkedZone lzone = new TLGSelDS.LinkedZone(new System.IO.FileStream(args[0], System.IO.FileMode.Open, System.IO.FileAccess.Read), f);
                                TLGSelDS.LinkedZone lzone = new TLGSelDS.LinkedZone(args[0], f);			
                */
                int[] basetkremaplist;
                int[] toptkremaplist;
                int[] bottomtkremaplist;

                SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment ai = null;
                SySal.OperaDb.Scanning.DBMIPMicroTrackIndex dbmi = null;
                SySal.Scanning.PostProcessing.SlopeCorrections sc = null;
                SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex bi = null;
                try
                {
                    ai = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment)SySal.OperaPersistence.Restore(args[0], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment));
                }
                catch (Exception)
                {
                    ai = null;
                }
                try
                {
                    bi = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex)SySal.OperaPersistence.Restore(args[0], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex));
                }
                catch (Exception)
                {
                    bi = null;
                }
                try
                {
                    dbmi = (SySal.OperaDb.Scanning.DBMIPMicroTrackIndex)SySal.OperaPersistence.Restore(args[0], typeof(SySal.OperaDb.Scanning.DBMIPMicroTrackIndex));
                }
                catch (Exception)
                {
                    dbmi = null;
                }
                try
                {
                    sc = (SySal.Scanning.PostProcessing.SlopeCorrections)SySal.OperaPersistence.Restore(args[0], typeof(SySal.Scanning.PostProcessing.SlopeCorrections));
                }
                catch (Exception)
                {
                    sc = null;
                }

                SySal.DataStreams.OPERALinkedZone olz = ((SySal.DataStreams.OPERALinkedZone)(new Exe().ProcessDataEx2(args[1], args[0], args[2], CleanMicroTracks, includelist, excludelist, out basetkremaplist, out toptkremaplist, out bottomtkremaplist)));
                if (olz != null)
                {
                    olz.Dispose();
                    olz = null;
                }
                
                if (args[1].ToLower().EndsWith(".tlg"))
                {
                    if (ai != null)
                    {
                        System.Collections.ArrayList zr = new System.Collections.ArrayList();
                        foreach (int iz in ai.Ids) zr.Add(iz);
                        zr.Sort();
                        System.Collections.ArrayList ar = new System.Collections.ArrayList();
                        int i;
                        for (i = 0; i < basetkremaplist.Length; i++)
                        {
                            if (zr.BinarySearch(basetkremaplist[i]) >= 0) ar.Add(i);
                        }
                        ai.Ids = (int[])ar.ToArray(typeof(int));
                        SySal.OperaPersistence.Persist(args[1], ai);
                    }
                    if (dbmi != null)
                    {
                        if (CleanMicroTracks)
                        {
                            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[] tix = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[toptkremaplist.Length];
                            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[] bix = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[bottomtkremaplist.Length];
                            int i;
                            for (i = 0; i < toptkremaplist.Length; i++)
                                tix[i] = dbmi.TopTracksIndex[toptkremaplist[i]];
                            for (i = 0; i < bottomtkremaplist.Length; i++)
                                bix[i] = dbmi.BottomTracksIndex[bottomtkremaplist[i]];
                            dbmi.TopTracksIndex = tix;
                            dbmi.BottomTracksIndex = bix;
                        }
                        SySal.OperaPersistence.Persist(args[1], dbmi);
                    }
                    if (sc != null)
                    {
                        SySal.OperaPersistence.Persist(args[1], sc);
                    }
                    if (bi != null)
                    {
                        int[] newbasetkremaplist = new int[basetkremaplist.Length];
                        int i;
                        for (i = 0; i < newbasetkremaplist.Length; i++)
                            newbasetkremaplist[i] = bi.Ids[basetkremaplist[i]];
                        bi.Ids = newbasetkremaplist;
                    }
                    else
                    {
                        bi = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex();
                        bi.Ids = basetkremaplist;
                    }
                    SySal.OperaPersistence.Persist(args[1], bi);
                }
#if (!(DEBUG))
            }
            catch (Exception x)
            {
                Console.Error.WriteLine(x);
            }
#endif
        }

        /// <summary>
        /// Processes data. Used to access TLGSelDS functions programmatically.
        /// </summary>
        /// <param name="lz">the linked zone to be selected.</param>
        /// <param name="selstring">the selection string.</param>
        /// <param name="cleanmicrotracks">if <c>true</c>, microtracks not attached to any surviving base tracks are deleted too.</param>
        /// <returns>a linked zone with the selected base tracks and microtracks.</returns>
        public SySal.Scanning.Plate.IO.OPERA.LinkedZone ProcessData(string outpath, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, string selstring, bool cleanmicrotracks)
        {
            int[] basetkremaplist;
            int[] toptkremaplist;
            int[] bottomtkremaplist;            
            TLGSelDS.LinkedZone.Select(outpath, lz, new CStyleParsedFunction(selstring), cleanmicrotracks, null, null, out basetkremaplist, out toptkremaplist, out bottomtkremaplist);
            return outpath.ToLower().EndsWith(".tlg") ? new SySal.DataStreams.OPERALinkedZone(outpath) : null;
        }

        /// <summary>
        /// Processes data. Used to access TLGSelDS functions programmatically. Extends <c>ProcessDataEx</c>.
        /// </summary>
        /// <param name="outpath">the path to the output TLG file.</param>
        /// <param name="inpath">the path to the input TLG file.</param>        
        /// <param name="selstring">the selection string.</param>
        /// <param name="cleanmicrotracks">if <c>true</c>, microtracks not attached to any surviving base tracks are deleted too.</param>
        /// <param name="includelist">list of Ids of base tracks to be included.</param>
        /// <param name="excludelist">list of Ids of base tracks to be excluded.</param>
        /// <param name="basetkremaplist">the Ids in this list allow mapping the base track Ids in the selected linked zone to the original base track Ids.</param>
        /// <param name="toptkremaplist">the Ids in this list allow mapping the top microtrack Ids in the selected linked zone to the original top microtrack Ids.</param>
        /// <param name="bottomtkremaplist">the Ids in this list allow mapping the bottom microtrack Ids in the selected linked zone to the original bottom microtrack Ids.</param>
        /// <returns>a linked zone with the selected base tracks and microtracks.</returns>
        public SySal.Scanning.Plate.IO.OPERA.LinkedZone ProcessDataEx2(string outpath, string inpath, string selstring, bool cleanmicrotracks, int[] includelist, int[] excludelist, out int[] basetkremaplist, out int[] toptkremaplist, out int[] bottomtkremaplist)
        {            
            SySal.DataStreams.OPERALinkedZone lz = new SySal.DataStreams.OPERALinkedZone(inpath);            
            TLGSelDS.LinkedZone.Select(outpath, lz, new CStyleParsedFunction(selstring), cleanmicrotracks, includelist, excludelist, out basetkremaplist, out toptkremaplist, out bottomtkremaplist);            
            lz.Dispose();            
            return outpath.ToLower().EndsWith(".tlg") ? new SySal.DataStreams.OPERALinkedZone(outpath) : null;
        }
    }
}
