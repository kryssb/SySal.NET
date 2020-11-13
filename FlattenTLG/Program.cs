using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.FlattenTLG
{
    class Exe
    {
        class RTrack
        {
            public SySal.BasicTypes.Vector2 Slope;
            public SySal.BasicTypes.Vector2 Position;
        }

        class RTrackCell
        {
            public SySal.BasicTypes.Rectangle Extents;
            public SySal.BasicTypes.Vector2 Center;
            public int Count;
            public SySal.BasicTypes.Vector2 Average;
            SySal.BasicTypes.Vector2 Sum;
            System.IO.FileStream TempStream;
            System.IO.BinaryWriter TempWrite;
            System.IO.BinaryReader TempRead;
            string FName;
            public SySal.DAQSystem.Scanning.IntercalibrationInfo AlignInfo;
            public SySal.BasicTypes.Vector2 SlopeAlignInfo;
            public int Matches;
            public NumericalTools.ComputationResult Result;
            public RTrackCell(SySal.BasicTypes.Rectangle rect, SySal.BasicTypes.Vector2 gencenter)
            {
                Extents = rect;
                Center.X = 0.5 * (rect.MinX + rect.MaxX);
                Center.Y = 0.5 * (rect.MinY + rect.MaxY);
                FName = System.Environment.GetEnvironmentVariable("TEMP");
                if (FName.EndsWith("/") == false && FName.EndsWith("\\") == false) FName += "/";
                FName += "flattentlg_" + System.Guid.NewGuid().ToString() + ".tmp";
                TempStream = new System.IO.FileStream(FName, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                Count = 0;
                TempWrite = new System.IO.BinaryWriter(TempStream);
                TempRead = new System.IO.BinaryReader(TempStream);
                Sum.X = 0.0;
                Sum.Y = 0.0;
                Average.X = 0.0;
                Average.Y = 0.0;
                AlignInfo.RX = Center.X; //gencenter.X;
                AlignInfo.RY = Center.Y; //gencenter.Y;
            }
            ~RTrackCell()
            {
                if (TempRead != null) TempRead.Close();
                if (TempWrite != null) TempWrite.Close();
                if (TempStream != null) TempWrite.Close();
                try
                {
                    System.IO.File.Delete(FName);
                }
                catch (Exception x) 
                {
                    Console.WriteLine("Exception while finalizing. File \"" + FName + "\".\r\n" + x.ToString());
                }
            }
            public void Add(RTrack rtr)
            {
                TempStream.Position = 32 * Count;
                TempWrite.Write(rtr.Slope.X);
                TempWrite.Write(rtr.Slope.Y);
                TempWrite.Write(rtr.Position.X);
                TempWrite.Write(rtr.Position.Y);
                Count++;
                Sum.X += rtr.Position.X;
                Sum.Y += rtr.Position.Y;
                Average.X = Sum.X / Count;
                Average.Y = Sum.Y / Count;
            }
            public RTrack Get(int i)
            {
                if (i < 0 || i >= Count) throw new Exception("Wrong index " + i + " (Count = " + Count + ")");
                TempStream.Position = 32 * i;
                RTrack rtr = new RTrack();
                rtr.Slope.X = TempRead.ReadDouble();
                rtr.Slope.Y = TempRead.ReadDouble();
                rtr.Position.X = TempRead.ReadDouble();
                rtr.Position.Y = TempRead.ReadDouble();
                return rtr;
            }
        }

        delegate double dSel(SySal.Scanning.MIPBaseTrack tk);

        class SelFunc
        {
            public string Name;
            public string Desc;
            public dSel Evaluate;
            public SelFunc(string name, string desc, dSel ev) { Name = name; Desc = desc; Evaluate = ev; }
        }

        static double fA(SySal.Scanning.MIPBaseTrack tk) { return (double)tk.Info.AreaSum; }
        static double fN(SySal.Scanning.MIPBaseTrack tk) { return (double)tk.Info.Count; }
        static double fS(SySal.Scanning.MIPBaseTrack tk) { return tk.Info.Sigma; }
        static double fSX(SySal.Scanning.MIPBaseTrack tk) { return tk.Info.Slope.X; }
        static double fSY(SySal.Scanning.MIPBaseTrack tk) { return tk.Info.Slope.Y; }

        static SelFunc[] KnownFunctions = new SelFunc[] 
        {
            new SelFunc("A", "Areasum", new dSel(fA)),
            new SelFunc("N", "Grains", new dSel(fN)),
            new SelFunc("S", "Sigma", new dSel(fS)),
            new SelFunc("SX", "X Slope", new dSel(fSX)),
            new SelFunc("SY", "Y Slope", new dSel(fSY))
        };        

        static System.Text.RegularExpressions.Regex rx_CellMap = new System.Text.RegularExpressions.Regex(@"\s*CELLMAP\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)");

        static System.Text.RegularExpressions.Regex rx_Cell = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)");

        static System.Text.RegularExpressions.Regex rx_XY = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)");

        static void Main(string[] args)
        {
            if (args.Length != 13 && args.Length != 4 && args.Length != 2)
            {
                Console.WriteLine("usage: FlattenTLG.exe <cell map> <TLG to be flattened> <output TLG> <min matches>");
                Console.WriteLine("   or");
                Console.WriteLine("usage: FlattenTLG.exe <cell map> <min matches>");
                Console.WriteLine("           (opens a console to query the transformation map generator)");
                Console.WriteLine("   or");
                Console.WriteLine("usage: FlattenTLG.exe <reference TLG (supposed flat)> <TLG to be flattened> <output TLG> <cell size> <slope tol> <pos tol> <pos sweep> <z projection> <selection string> <min matches> <z adjust> <z step> <parallel (true|false)>");
                Console.WriteLine("Selection function variables:");
                foreach (SelFunc sf in KnownFunctions)
                    Console.WriteLine(sf.Name + " -> " + sf.Desc);
                return;
            }
            bool usereadymap = (args.Length < 13);
            bool useconsole = (args.Length == 2);
            string reftlg = args[0];
            string worktlg = useconsole ? "" : args[1];
            string outtlg = useconsole ? "" : args[2];
            uint MinMatches = 0;
            int xcells = 0;
            int ycells = 0;
            double cellsize = 0.0;
            int ix, iy;
            int i, j, k;
            SySal.BasicTypes.Vector2 Center = new SySal.BasicTypes.Vector2();
            SySal.BasicTypes.Rectangle WorkRect;
            SySal.Scanning.Plate.IO.OPERA.LinkedZone worklz = null;
            if (useconsole == false)
            {
                worklz = SySal.DataStreams.OPERALinkedZone.FromFile(worktlg);
                WorkRect = worklz.Extents;
            }
            else WorkRect = new SySal.BasicTypes.Rectangle();            
            SySal.BasicTypes.Rectangle RefRect = new SySal.BasicTypes.Rectangle();
            RTrackCell[,] WorkCells;
            if (usereadymap)
            {
                MinMatches = Convert.ToUInt32( args[useconsole ? 1 : 3] );
                System.IO.StreamReader cr = new System.IO.StreamReader(args[0]);
                while (cr.EndOfStream == false)
                {
                    System.Text.RegularExpressions.Match m = rx_CellMap.Match(cr.ReadLine());
                    if (m.Success)
                    {
                        RefRect.MinX = Convert.ToDouble(m.Groups[1].Value);
                        RefRect.MaxX = Convert.ToDouble(m.Groups[2].Value);
                        RefRect.MinY = Convert.ToDouble(m.Groups[3].Value);
                        RefRect.MaxY = Convert.ToDouble(m.Groups[4].Value);
                        cellsize = Convert.ToDouble(m.Groups[5].Value);
                        xcells = Convert.ToInt32(m.Groups[6].Value);
                        ycells = Convert.ToInt32(m.Groups[7].Value);
                        break;
                    }
                }
                Center.X = 0.5 * (RefRect.MinX + RefRect.MaxX);
                Center.Y = 0.5 * (RefRect.MinY + RefRect.MaxY);
                WorkCells = new RTrackCell[xcells, ycells];
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                    {
                        SySal.BasicTypes.Rectangle rect = new SySal.BasicTypes.Rectangle();
                        rect.MinX = RefRect.MinX + ix * cellsize;
                        rect.MaxX = rect.MinX + cellsize;
                        rect.MinY = RefRect.MinY + iy * cellsize;
                        rect.MaxY = rect.MinY + cellsize;
                        WorkCells[ix, iy] = new RTrackCell(rect, Center);
                        WorkCells[ix, iy].Result = NumericalTools.ComputationResult.InvalidInput;
                    }
                while (cr.EndOfStream == false)
                {
                    System.Text.RegularExpressions.Match m = rx_Cell.Match(cr.ReadLine());
                    if (m.Success)
                    {
                        ix = Convert.ToInt32(m.Groups[1].Value);
                        iy = Convert.ToInt32(m.Groups[2].Value);
                        WorkCells[ix, iy].Result = NumericalTools.ComputationResult.OK;
                        WorkCells[ix, iy].Matches = Convert.ToInt32(m.Groups[3].Value);
                        WorkCells[ix, iy].Average.X = Convert.ToDouble(m.Groups[4].Value);
                        WorkCells[ix, iy].Average.Y = Convert.ToDouble(m.Groups[5].Value);
                        WorkCells[ix, iy].AlignInfo.MXX = Convert.ToDouble(m.Groups[6].Value);
                        WorkCells[ix, iy].AlignInfo.MXY = Convert.ToDouble(m.Groups[7].Value);
                        WorkCells[ix, iy].AlignInfo.MYX = Convert.ToDouble(m.Groups[8].Value);
                        WorkCells[ix, iy].AlignInfo.MYY = Convert.ToDouble(m.Groups[9].Value);
                        WorkCells[ix, iy].AlignInfo.TX = Convert.ToDouble(m.Groups[10].Value);
                        WorkCells[ix, iy].AlignInfo.TY = Convert.ToDouble(m.Groups[11].Value);
                        WorkCells[ix, iy].AlignInfo.TZ = Convert.ToDouble(m.Groups[12].Value);
                        WorkCells[ix, iy].SlopeAlignInfo.X = Convert.ToDouble(m.Groups[13].Value);
                        WorkCells[ix, iy].SlopeAlignInfo.Y = Convert.ToDouble(m.Groups[14].Value);                        
                    }
                }
                cr.Close();
                if (useconsole)
                {
                    GridInterpolation G1 = new GridInterpolation(WorkCells, cellsize, RefRect, (int)MinMatches);
                    Console.WriteLine("Type a pair of coordinates ( X Y ) to get the transformation map.\r\nEOF (CTRL+Z to exit).");
                    double x, y;
                    string line;
                    while ((line = Console.ReadLine()) != null)
                    {
                        System.Text.RegularExpressions.Match m = rx_XY.Match(line);
                        x = Convert.ToDouble(m.Groups[1].Value);
                        y = Convert.ToDouble(m.Groups[2].Value);
                        SySal.BasicTypes.Vector dslope = new SySal.BasicTypes.Vector();
                        SySal.DAQSystem.Scanning.IntercalibrationInfo dpos = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                        bool result = G1.Evaluate(x, y, ref dslope, ref dpos);
                        Console.WriteLine(x + " " + y + " -> " + (result ? "OK" : "FAILED") + " " + dpos.RX + " " + dpos.RY + " " + dpos.MXX + " " + dpos.MXY + " " + dpos.MYX + " " + dpos.MYY + " " + dpos.TX + " " + dpos.TY + " " + dpos.TZ + " " + dslope.X + " " + dslope.Y);
                    }
                    return;
                }
            }
            else
            {
                cellsize = Convert.ToDouble(args[3]);
                double slopetol = Convert.ToDouble(args[4]);
                double postol = Convert.ToDouble(args[5]);
                double possweep = Convert.ToDouble(args[6]);
                double DZ = Convert.ToDouble(args[7]);
                string selstring = args[8];
                MinMatches = Convert.ToUInt32(args[9]);
                double ZAdj = Convert.ToDouble(args[10]);
                double ZStep = Convert.ToDouble(args[11]);
                bool IsParallel = Convert.ToBoolean(args[12]);
                NumericalTools.CStyleParsedFunction S = new NumericalTools.CStyleParsedFunction(selstring);
                dSel[] pMap = new dSel[S.ParameterList.Length];            
                for (j = 0; j < S.ParameterList.Length; j++)
                {
                    string sp = S.ParameterList[j];
                    for (i = 0; i < KnownFunctions.Length && String.Compare(sp, KnownFunctions[i].Name, true) != 0; i++) ;
                    if (i == KnownFunctions.Length)
                        throw new Exception("Unknown parameter \"" + sp + "\".");
                    pMap[j] = KnownFunctions[i].Evaluate;
                }
                SySal.Scanning.Plate.IO.OPERA.LinkedZone reflz = SySal.DataStreams.OPERALinkedZone.FromFile(reftlg);
                RefRect = reflz.Extents;
                if (WorkRect.MinX > RefRect.MinX) RefRect.MinX = WorkRect.MinX;
                if (WorkRect.MaxX < RefRect.MaxX) RefRect.MaxX = WorkRect.MaxX;
                if (WorkRect.MinY > RefRect.MinY) RefRect.MinY = WorkRect.MinY;
                if (WorkRect.MaxY < RefRect.MaxY) RefRect.MaxY = WorkRect.MaxY;
                Center.X = 0.5 * (RefRect.MinX + RefRect.MaxX);
                Center.Y = 0.5 * (RefRect.MinY + RefRect.MaxY);
                xcells = (int)Math.Ceiling((RefRect.MaxX - RefRect.MinX) / cellsize);
                ycells = (int)Math.Ceiling((RefRect.MaxY - RefRect.MinY) / cellsize);
                Console.WriteLine("X/Y Cells: " + xcells + "/" + ycells);
                if (xcells <= 0 || ycells <= 0) throw new Exception("Null working area.");                
                RTrackCell[,] RefCells = new RTrackCell[xcells, ycells];
                WorkCells = new RTrackCell[xcells, ycells];
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                    {
                        SySal.BasicTypes.Rectangle rect = new SySal.BasicTypes.Rectangle();
                        rect.MinX = RefRect.MinX + ix * cellsize;
                        rect.MaxX = rect.MinX + cellsize;
                        rect.MinY = RefRect.MinY + iy * cellsize;
                        rect.MaxY = rect.MinY + cellsize;
                        RefCells[ix, iy] = new RTrackCell(rect, Center);
                        WorkCells[ix, iy] = new RTrackCell(rect, Center);
                    }
                SySal.Scanning.Plate.IO.OPERA.LinkedZone lz;
                RTrackCell[,] rtc;
                for (i = 0; i < 2; i++)
                {
                    if (i == 0)
                    {
                        lz = reflz;
                        rtc = RefCells;
                    }
                    else
                    {
                        lz = worklz;
                        rtc = WorkCells;
                    }
                    for (j = 0; j < lz.Length; j++)
                    {
                        SySal.Scanning.MIPBaseTrack tk = lz[j] as SySal.Scanning.MIPBaseTrack;
                        for (k = 0; k < pMap.Length; k++)
                            S[k] = pMap[k](tk);
                        if (S.Evaluate() != 0.0)
                        {
                            ix = (int)((tk.Info.Intercept.X - RefRect.MinX) / cellsize);
                            iy = (int)((tk.Info.Intercept.Y - RefRect.MinY) / cellsize);
                            if (ix >= 0 && ix < xcells && iy >= 0 && iy < ycells)
                            {
                                RTrack rtr = new RTrack();
                                rtr.Slope.X = tk.Info.Slope.X;
                                rtr.Slope.Y = tk.Info.Slope.Y;
                                rtr.Position.X = tk.Info.Intercept.X;
                                rtr.Position.Y = tk.Info.Intercept.Y;
                                rtc[ix, iy].Add(rtr);
                            }
                        }
                    }
                }
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                    {
                        Console.WriteLine("Ref " + RefCells[ix, iy].Average.X + " " + RefCells[ix, iy].Average.Y + " " + RefCells[ix, iy].Count);
                        Console.WriteLine("Work " + WorkCells[ix, iy].Average.X + " " + WorkCells[ix, iy].Average.Y + " " + WorkCells[ix, iy].Count);
                    }
                SySal.Processing.QuickMapping.QuickMapper QM = new SySal.Processing.QuickMapping.QuickMapper();
                SySal.Processing.QuickMapping.Configuration qmc = QM.Config as SySal.Processing.QuickMapping.Configuration;
                qmc.FullStatistics = false;
                qmc.UseAbsoluteReference = true;
                qmc.PosTol = postol;
                qmc.SlopeTol = slopetol;
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo[] rinfo = new SySal.Tracking.MIPEmulsionTrackInfo[RefCells[ix, iy].Count];
                        SySal.Tracking.MIPEmulsionTrackInfo[] winfo = new SySal.Tracking.MIPEmulsionTrackInfo[WorkCells[ix, iy].Count];
                        for (i = 0; i < 2; i++)
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo[] inf = (i == 0) ? rinfo : winfo;
                            RTrackCell[,] cells = (i == 0) ? RefCells : WorkCells;
                            double dz = (i == 0) ? 0.0 : DZ;
                            for (j = 0; j < inf.Length; j++)
                            {
                                RTrack r = cells[ix, iy].Get(j);
                                inf[j] = new SySal.Tracking.MIPEmulsionTrackInfo();
                                inf[j].Slope.X = r.Slope.X;
                                inf[j].Slope.Y = r.Slope.Y;
                                inf[j].Intercept.X = r.Position.X;
                                inf[j].Intercept.Y = r.Position.Y;
                                inf[j].Intercept.Z = dz;
                            }
                        }
                        SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] pairs = new SySal.Scanning.PostProcessing.PatternMatching.TrackPair[0];
                        double bestdz = 0.0;
                        if (rinfo.Length >= 2 && winfo.Length >= 2)
                        {
                            double dz1;
                            if (IsParallel)
                            {
                                System.Collections.ArrayList thrarr = new System.Collections.ArrayList();
                                for (dz1 = -ZAdj; dz1 <= ZAdj; dz1 += ZStep)
                                {
                                    MapThread mthr = new MapThread();
                                    mthr.m_rinfo = rinfo;
                                    mthr.m_winfo = winfo;
                                    mthr.m_DZ = DZ + dz1;
                                    mthr.m_PosSweep = possweep;
                                    mthr.m_PosTol = postol;
                                    mthr.m_SlopeTol = slopetol;
                                    mthr.m_Thread = new System.Threading.Thread(new System.Threading.ThreadStart(mthr.Execute));
                                    mthr.m_Thread.Start();
                                    thrarr.Add(mthr);
                                }
                                foreach (MapThread mt in thrarr)
                                {
                                    mt.m_Thread.Join();
                                    if (mt.m_Pairs.Length > pairs.Length)
                                    {
                                        bestdz = mt.m_DZ - DZ;
                                        pairs = mt.m_Pairs;
                                    }
                                }
                            }
                            else
                                for (dz1 = -ZAdj; dz1 <= ZAdj; dz1 += ZStep)
                                {
                                    SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] qairs = QM.Match(rinfo, winfo, DZ + dz1, possweep, possweep);
                                    if (qairs.Length > pairs.Length)
                                    {
                                        bestdz = dz1;
                                        pairs = qairs;
                                    }
                                }
                        }
                        double[] alignpars = new double[7];
                        SySal.BasicTypes.Vector2 slopedelta = new SySal.BasicTypes.Vector2();
                        SySal.BasicTypes.Vector2 slopetolv = new SySal.BasicTypes.Vector2();
                        double[] dslx = new double[pairs.Length];
                        double[] dsly = new double[pairs.Length];
                        for (j = 0; j < pairs.Length; j++)
                            dslx[j] = pairs[j].First.Info.Slope.X - pairs[j].Second.Info.Slope.X;
                        PeakFit(dslx, slopetol, out slopedelta.X, out slopetolv.X);
                        for (j = 0; j < pairs.Length; j++)
                            dsly[j] = pairs[j].First.Info.Slope.Y - pairs[j].Second.Info.Slope.Y;
                        PeakFit(dsly, slopetol, out slopedelta.Y, out slopetolv.Y);
                        int gooddslopes = 0;
                        for (j = 0; j < pairs.Length; j++)
                            if ((slopedelta.X - slopetolv.X) < dslx[j] && dslx[j] < (slopedelta.X + slopetolv.X) && (slopedelta.Y - slopetolv.Y) < dsly[j] && dsly[j] < (slopedelta.Y + slopetolv.Y))
                                gooddslopes++;
                        if (gooddslopes > 0)
                        {
                            double[] DX = new double[gooddslopes];
                            double[] DY = new double[gooddslopes];
                            double[] X = new double[gooddslopes];
                            double[] Y = new double[gooddslopes];
                            double[] SX = new double[gooddslopes];
                            double[] SY = new double[gooddslopes];
                            for (j = i = 0; j < pairs.Length; j++)
                                if ((slopedelta.X - slopetolv.X) < dslx[j] && dslx[j] < (slopedelta.X + slopetolv.X) && (slopedelta.Y - slopetolv.Y) < dsly[j] && dsly[j] < (slopedelta.Y + slopetolv.Y))
                                {
                                    X[i] = pairs[j].Second.Info.Intercept.X - WorkCells[ix, iy].AlignInfo.RX;
                                    Y[i] = pairs[j].Second.Info.Intercept.Y - WorkCells[ix, iy].AlignInfo.RY;
                                    SX[i] = pairs[j].Second.Info.Slope.X;
                                    SY[i] = pairs[j].Second.Info.Slope.Y;
                                    DX[i] = pairs[j].First.Info.Intercept.X - pairs[j].Second.Info.Intercept.X;
                                    DY[i] = pairs[j].First.Info.Intercept.Y - pairs[j].Second.Info.Intercept.Y;
                                    //System.IO.File.AppendAllText(@"c:\flattentlg.txt", "\r\n" + ix + " " + iy + " " + i + " " + j + " " + pairs.Length + " " + gooddslopes + " " + WorkCells[ix, iy].AlignInfo.RX + " " + WorkCells[ix, iy].AlignInfo.RY + " " + X[i] + " " + Y[i] + " " + SX[i] + " " + SY[i] + "  " + DX[i] + " " + DY[i] + " " + bestdz);
                                    i++;
                                }
                            WorkCells[ix, iy].Result = IteratedAffineFocusing(DX, DY, X, Y, SX, SY, postol, ref alignpars);
                        }
                        else WorkCells[ix, iy].Result = NumericalTools.ComputationResult.InvalidInput;
                        WorkCells[ix, iy].Matches = pairs.Length;
                        WorkCells[ix, iy].AlignInfo.TZ = alignpars[6] + bestdz;
                        WorkCells[ix, iy].AlignInfo.TX = alignpars[4];
                        WorkCells[ix, iy].AlignInfo.TY = alignpars[5];
                        WorkCells[ix, iy].AlignInfo.MXX = 1.0 + alignpars[0];
                        WorkCells[ix, iy].AlignInfo.MXY = 0.0 + alignpars[1];
                        WorkCells[ix, iy].AlignInfo.MYX = 0.0 + alignpars[2];
                        WorkCells[ix, iy].AlignInfo.MYY = 1.0 + alignpars[3];
                        WorkCells[ix, iy].SlopeAlignInfo = slopedelta;
                        Console.WriteLine("Fit " + WorkCells[ix, iy].Result + " " + WorkCells[ix, iy].AlignInfo.MXX + " " + WorkCells[ix, iy].AlignInfo.MXY + " " + WorkCells[ix, iy].AlignInfo.MYX + " " + WorkCells[ix, iy].AlignInfo.MYY + " " + WorkCells[ix, iy].AlignInfo.TX + " " + WorkCells[ix, iy].AlignInfo.TY + " " + WorkCells[ix, iy].AlignInfo.TZ + " " + WorkCells[ix, iy].SlopeAlignInfo.X + " " + WorkCells[ix, iy].SlopeAlignInfo.Y);
                    }
                int goodcells = 0;
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                        if (WorkCells[ix, iy].Result == NumericalTools.ComputationResult.OK && WorkCells[ix, iy].Matches >= MinMatches)
                            goodcells++;
                Console.WriteLine("Good cells: " + goodcells);

                Console.WriteLine("--------CELLS");
                Console.WriteLine("CELLMAP " + RefRect.MinX + " " + RefRect.MaxX + " " + RefRect.MinY + " " + RefRect.MaxY + " " + cellsize + " " + xcells + " " + ycells);
                Console.WriteLine("IX\tIY\tN\tX\tY\tMXX\tMXY\tMYX\tMYY\tTX\tTY\tTZ\tTDSX\tTDSY");
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                        if (WorkCells[ix, iy].Result == NumericalTools.ComputationResult.OK && WorkCells[ix, iy].Matches >= MinMatches)
                            Console.WriteLine(ix + "\t" + iy + "\t" + WorkCells[ix, iy].Matches + "\t" + WorkCells[ix, iy].Average.X + "\t" + WorkCells[ix, iy].Average.Y + "\t" + WorkCells[ix, iy].AlignInfo.MXX + "\t" + WorkCells[ix, iy].AlignInfo.MXY + "\t" + WorkCells[ix, iy].AlignInfo.MYX + "\t" + WorkCells[ix, iy].AlignInfo.MYY + "\t" + WorkCells[ix, iy].AlignInfo.TX + "\t" + WorkCells[ix, iy].AlignInfo.TY + "\t" + WorkCells[ix, iy].AlignInfo.TZ + "\t" + WorkCells[ix, iy].SlopeAlignInfo.X + "\t" + WorkCells[ix, iy].SlopeAlignInfo.Y);
                Console.WriteLine("--------ENDCELLS");
            }
            SySal.DataStreams.OPERALinkedZone.Writer outlzw = new SySal.DataStreams.OPERALinkedZone.Writer(outtlg, worklz.Id, worklz.Extents, worklz.Center, worklz.Transform);
            outlzw.SetZInfo(worklz.Top.TopZ, worklz.Top.BottomZ, worklz.Bottom.TopZ, worklz.Bottom.BottomZ);
            for (i = 0; i < ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)worklz.Top).ViewCount; i++)
                outlzw.AddView(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)worklz.Top).View(i), true);
            for (i = 0; i < ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)worklz.Bottom).ViewCount; i++)
                outlzw.AddView(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)worklz.Bottom).View(i), false);
            SySal.BasicTypes.Vector proj = new SySal.BasicTypes.Vector();
            Console.WriteLine("Writing flattened TLG...");
            GridInterpolation G = new GridInterpolation(WorkCells, cellsize, RefRect, (int)MinMatches);
            System.DateTime start = System.DateTime.Now;
            for (i = 0; i < worklz.Length; i++)
            {
                if (i % 1000 == 0)
                {
                    System.DateTime nw = System.DateTime.Now;
                    if ((nw - start).TotalMilliseconds >= 10000)
                    {
                        Console.WriteLine((i * 100 / worklz.Length) + "%");
                        start = nw;
                    }
                }
                SySal.Tracking.MIPEmulsionTrackInfo baseinfo = worklz[i].Info;
                SySal.DAQSystem.Scanning.IntercalibrationInfo transforminfo = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                transforminfo.RX = Center.X;
                transforminfo.RY = Center.Y;
                SySal.BasicTypes.Vector tds = new SySal.BasicTypes.Vector();

                G.Evaluate(baseinfo.Intercept.X, baseinfo.Intercept.Y, ref tds, ref transforminfo);

                proj.X = -baseinfo.Slope.X * transforminfo.TZ;
                proj.Y = -baseinfo.Slope.Y * transforminfo.TZ;
                proj.Z = 0.0;
                baseinfo.Intercept = transforminfo.Transform(baseinfo.Intercept) + proj;
                baseinfo.Slope = transforminfo.Deform(baseinfo.Slope) + tds;
                SySal.Scanning.MIPIndexedEmulsionTrack toptk = worklz[i].Top;
                SySal.Tracking.MIPEmulsionTrackInfo topinfo = toptk.Info;
                SySal.Scanning.MIPIndexedEmulsionTrack bottomtk = worklz[i].Bottom;
                SySal.Tracking.MIPEmulsionTrackInfo bottominfo = bottomtk.Info;
                topinfo.Intercept = transforminfo.Transform(topinfo.Intercept) + proj;
                topinfo.Slope = transforminfo.Deform(topinfo.Slope) + tds;
                bottominfo.Intercept = transforminfo.Transform(bottominfo.Intercept) + proj;
                bottominfo.Slope = transforminfo.Deform(bottominfo.Slope) + tds;
                outlzw.AddMIPEmulsionTrack(topinfo, i, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)toptk).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)toptk).OriginalRawData, true);
                outlzw.AddMIPEmulsionTrack(bottominfo, i, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)bottomtk).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)bottomtk).OriginalRawData, false);
                outlzw.AddMIPBasetrack(baseinfo, i, i, i);
            }            
            outlzw.Complete();
            Console.WriteLine("Written \"" + outtlg + "\".");
        }

        private static NumericalTools.ComputationResult IteratedAffineFocusing(double[] DX, double[] DY, double[] X, double[] Y, double[] SX, double[] SY, double postol, ref double[] alignpars)
        {
            int iteration, i, j;
            double xtol = postol;
            double ytol = postol;
            NumericalTools.ComputationResult res = NumericalTools.ComputationResult.SingularityEncountered;
            for (iteration = 0; iteration < 3; iteration++)
            {
                double[] mDX;
                double[] mDY;
                double[] mSX;
                double[] mSY;
                double[] mX;
                double[] mY;
                if (iteration == 0)
                {
                    mDX = DX;
                    mDY = DY;
                    mSX = SX;
                    mSY = SY;
                    mX = X;
                    mY = Y;
                }
                else
                {
                    double dx, dx2sum, dy, dy2sum;
                    dx2sum = dy2sum = 0.0;
                    bool[] take = new bool[DX.Length];
                    for (i = j = 0; i < take.Length; i++)
                    {
                        if (Math.Abs(dx = (DX[i] - alignpars[0] * X[i] - alignpars[1] * Y[i] - alignpars[4] - alignpars[6] * SX[i])) < xtol &&
                            Math.Abs(dy = (DY[i] - alignpars[2] * X[i] - alignpars[3] * Y[i] - alignpars[5] - alignpars[6] * SY[i])) < ytol)
                        {
                            dx2sum += dx * dx;
                            dy2sum += dy * dy;
                            take[i] = true;
                            j++;
                        }
                    }
                    if (j < 2) return NumericalTools.ComputationResult.SingularityEncountered;
                    xtol = 2.0 * Math.Sqrt(dx2sum / (j - 1));
                    ytol = 2.0 * Math.Sqrt(dy2sum / (j - 1));
                    mDX = new double[j];
                    mDY = new double[j];
                    mSX = new double[j];
                    mSY = new double[j];
                    mX = new double[j];
                    mY = new double[j];
                    for (i = j = 0; i < take.Length; i++)
                        if (take[i])
                        {
                            mDX[j] = DX[i];
                            mDY[j] = DY[i];
                            mSX[j] = SX[i];
                            mSY[j] = SY[i];
                            mX[j] = X[i];
                            mY[j] = Y[i];
                            j++;
                        }
                }
                res = NumericalTools.Fitting.Affine_Focusing(mDX, mDY, mX, mY, mSX, mSY, ref alignpars);
            }
            return res;
        }

        static void PeakFit(double[] x, double starttol, out double mean, out double tol)
        {
            mean = 0.0;
            tol = starttol;
            int i, j;
            for (i = 0; i < 3; i++)
            {
                double m = 0.0;
                double s = 0.0;
                j = 0;
                foreach (double xv in x)
                    if (mean - tol < xv && xv < mean + tol)
                    {
                        m += xv;
                        s += xv * xv;
                        j++;
                    }
                if (j <= 0) return;
                m /= j;
                s = Math.Sqrt(s / j - m * m);
                mean = m;
                tol = 2.0 * s;
            }
        }

        class GridInterpolation
        {
            RTrackCell[,] m_Grid;
            RTrackCell[,] m_Grid2;
            double CellSize;
            SySal.BasicTypes.Rectangle RefRect;
            int XCells, YCells;
            int MinMatches;
            int MaxXYCells;
            static int[][] dirs = new int[][] { new int[] { 0, 0, -1, -1 }, new int[] { 1, 0, 1, -1 }, new int[] { 1, 1, 1, 1 }, new int[] { 0, 1, -1, 1 } };
            public GridInterpolation(RTrackCell [,] g, double cellsize, SySal.BasicTypes.Rectangle refrect, int minmatches)
            {
                m_Grid = g;
                CellSize = cellsize;
                RefRect = refrect;
                SySal.BasicTypes.Vector2 c = new SySal.BasicTypes.Vector2();
                c.X = 0.5 * (refrect.MinX + RefRect.MaxX);
                c.Y = 0.5 * (refrect.MinY + RefRect.MaxY);
                XCells = m_Grid.GetLength(0);
                YCells = m_Grid.GetLength(1);
                MinMatches = minmatches;
                MaxXYCells = Math.Max(XCells, YCells);
                m_Grid2 = new RTrackCell[XCells, YCells];
                int ix, iy;
                for (ix = 0; ix < XCells; ix++)
                    for (iy = 0; iy < YCells; iy++)
                    {
                        RTrackCell rrc = m_Grid[ix, iy];
                        RTrackCell rtc = new RTrackCell(rrc.Extents, c);
                        m_Grid2[ix, iy] = rtc;
                        m_Grid2[ix, iy].Average = rtc.Center;
                        rtc.Result = NumericalTools.ComputationResult.OK;
                        rtc.Matches = Math.Max(minmatches, rrc.Matches);
                        SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
                        EvaluateW(rtc.Center.X, rtc.Center.Y, ref v, ref rtc.AlignInfo);
                        rtc.SlopeAlignInfo.X = v.X;
                        rtc.SlopeAlignInfo.Y = v.Y;
                    }
            }
            public bool EvaluateW(double x, double y, ref SySal.BasicTypes.Vector sloped, ref SySal.DAQSystem.Scanning.IntercalibrationInfo posd)
            {
                int ix = (int)((x - RefRect.MinX) / CellSize - 0.5);
                int iy = (int)((y - RefRect.MinY) / CellSize - 0.5);
                int iix, iiy;
                int i, j;
                double dx, dy;
                System.Collections.ArrayList arr = new System.Collections.ArrayList();
                if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells && m_Grid[ix, iy].Result == NumericalTools.ComputationResult.OK && m_Grid[ix, iy].Matches >= MinMatches)
                    arr.Add(m_Grid[ix, iy]);
                foreach (int [] dv in dirs)                
                    for (i = 0; i < MaxXYCells; i++)
                    {
                        iix = ix + dv[0] + i * dv[2];
                        if (iix < 0 || iix >= XCells) continue;
                        iiy = iy + dv[1] + i * dv[3];
                        if (iiy < 0 || iiy >= YCells) continue;
                        if (m_Grid[iix, iiy].Result == NumericalTools.ComputationResult.OK && m_Grid[iix, iiy].Matches >= MinMatches)
                        {
                            arr.Add(m_Grid[iix, iiy]);
                            break;
                        }
                    }
                RTrackCell[] activecells = (RTrackCell[])arr.ToArray(typeof(RTrackCell));
                if (activecells.Length == 0) return false;
                double[] w = new double[activecells.Length];
                double[] mw = new double[activecells.Length];
                double w_all = 0.0;
                for (i = 0; i < activecells.Length; i++)
                {
                    dx = x - activecells[i].Average.X;
                    dy = y - activecells[i].Average.Y;
                    mw[i] = dx * dx + dy * dy;                    
                }    
                for (i = 0; i < w.Length; i++)
                {
                    w[i] = 1.0;
                    for (j = 0; j < mw.Length; j++)
                        if (i != j) 
                            w[i] *= mw[j];
                }
                for (i = 0; i < w.Length; i++)
                    w_all += w[i];
                for (i = 0; i < w.Length; i++)
                    w[i] /= w_all;
                posd.MXX = posd.MXY = posd.MYX = posd.MYY = posd.TX = posd.TY = posd.TZ = 0.0;
                sloped.X = sloped.Y = 0;
                for (i = 0; i < w.Length; i++)
                {
                    posd.MXX += activecells[i].AlignInfo.MXX * w[i];
                    posd.MXY += activecells[i].AlignInfo.MXY * w[i];
                    posd.MYX += activecells[i].AlignInfo.MYX * w[i];
                    posd.MYY += activecells[i].AlignInfo.MYY * w[i];
                    posd.TX += (activecells[i].AlignInfo.TX + activecells[i].AlignInfo.MXX * (x - activecells[i].AlignInfo.RX) + activecells[i].AlignInfo.MXY * (y - activecells[i].AlignInfo.RY) + (activecells[i].AlignInfo.RX - x)) * w[i];
                    posd.TY += (activecells[i].AlignInfo.TY + activecells[i].AlignInfo.MYX * (x - activecells[i].AlignInfo.RX) + activecells[i].AlignInfo.MYY * (y - activecells[i].AlignInfo.RY) + (activecells[i].AlignInfo.RY - y)) * w[i];
                    posd.TZ += activecells[i].AlignInfo.TZ * w[i];
                    sloped.X += activecells[i].SlopeAlignInfo.X * w[i];
                    sloped.Y += activecells[i].SlopeAlignInfo.Y * w[i];
                }
                sloped.Z = 0.0;
                posd.RX = x;
                posd.RY = y;
                return true;    
            }
            public bool Evaluate(double x, double y, ref SySal.BasicTypes.Vector sloped, ref SySal.DAQSystem.Scanning.IntercalibrationInfo posd)
            {
                double tx = (x - RefRect.MinX) / CellSize - 0.5;
                double ty = (y - RefRect.MinY) / CellSize - 0.5;
                int ix = (int)tx;
                int iy = (int)ty;
                double mux = tx - ix;
                double muy = ty - iy;
                double[,] w = new double[,] { { (1.0 - muy) * (1.0 - mux), (1.0 - muy) * mux }, { muy * (1.0 - mux), muy * mux } };
                int itx, ity;
                posd.MXX = posd.MXY = posd.MYX = posd.MYY = posd.TX = posd.TY = posd.TZ = 0.0;
                sloped.X = sloped.Y = 0;
                for (itx = 0; itx <= 1; itx++)
                {
                    int iix = ix + itx;
                    if (iix < 0) iix = 0;
                    else if (iix >= XCells) iix = XCells - 1;
                    for (ity = 0; ity <= 1; ity++)
                    {
                        int iiy = iy + ity;
                        if (iiy < 0) iiy = 0;
                        else if (iiy >= YCells) iiy = YCells - 1;
                        double tw = w[ity, itx];
                        RTrackCell g = m_Grid2[iix, iiy];

                        posd.MXX += g.AlignInfo.MXX * tw;
                        posd.MXY += g.AlignInfo.MXY * tw;
                        posd.MYX += g.AlignInfo.MYX * tw;
                        posd.MYY += g.AlignInfo.MYY * tw;
                        posd.TX += (g.AlignInfo.TX + g.AlignInfo.MXX * (x - g.AlignInfo.RX) + g.AlignInfo.MXY * (y - g.AlignInfo.RY) + (g.AlignInfo.RX - x)) * tw;
                        posd.TY += (g.AlignInfo.TY + g.AlignInfo.MYX * (x - g.AlignInfo.RX) + g.AlignInfo.MYY * (y - g.AlignInfo.RY) + (g.AlignInfo.RY - y)) * tw;
                        posd.TZ += g.AlignInfo.TZ * tw;
                        sloped.X += g.SlopeAlignInfo.X * tw;
                        sloped.Y += g.SlopeAlignInfo.Y * tw;
                    }
                }
                sloped.Z = 0.0;
                posd.RX = x;
                posd.RY = y;
                return true;
            }
        }

        class MapThread
        {
            public SySal.Tracking.MIPEmulsionTrackInfo[] m_rinfo;
            public SySal.Tracking.MIPEmulsionTrackInfo[] m_winfo;
            public double m_DZ;
            public double m_PosSweep;
            public double m_PosTol;
            public double m_SlopeTol;            
            public SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] m_Pairs;
            public void Execute()
            {
                SySal.Processing.QuickMapping.QuickMapper QM = new SySal.Processing.QuickMapping.QuickMapper();
                SySal.Processing.QuickMapping.Configuration C = QM.Config as SySal.Processing.QuickMapping.Configuration;
                C.FullStatistics = false;
                C.Name = "";
                C.PosTol = m_PosTol;
                C.SlopeTol = m_SlopeTol;
                C.UseAbsoluteReference = true;
                m_Pairs = QM.Match(m_rinfo, m_winfo, m_DZ, m_PosSweep, m_PosSweep);
            }
            public System.Threading.Thread m_Thread;
        }
    }
}
