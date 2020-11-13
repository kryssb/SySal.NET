using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;

namespace SySal.Executables.TLGCleanDS
{
    public enum SelectionMode { LowerSigma, MoreGrains }

    public class Exe
    {
        public class TLGCleanConfig
        {
            public double PosTol = 10;
            public double SlopeTol = 0.03;
            public SelectionMode SelMode = SelectionMode.LowerSigma;
        }

        public class BufferedLinkedZone
        {
            public SySal.DataStreams.OPERALinkedZone _LZ;
            const int _MaxCapacity = 10000;
            Dictionary<int, Tuple<int, long, SySal.Scanning.MIPBaseTrack>> _Buffer = new Dictionary<int, Tuple<int, long, Scanning.MIPBaseTrack>>(_MaxCapacity);
            long _Access = 0;
            public SySal.Scanning.MIPBaseTrack this[int index]
            {
                get
                {
                    Tuple<int, long, SySal.Scanning.MIPBaseTrack> v = null;
                    if (_Buffer.TryGetValue(index, out v))
                    {
                        _Buffer[index] = new Tuple<int, long, Scanning.MIPBaseTrack>(v.Item1, ++_Access, v.Item3);
                        return v.Item3;
                    }
                    SySal.Scanning.MIPBaseTrack t;
                    lock (_LZ)
                        t = _LZ[index];
                    _Buffer[index] = new Tuple<int, long, SySal.Scanning.MIPBaseTrack>(index, ++_Access, t);
                    if (_Buffer.Count >= _MaxCapacity)
                    {
                        var temp = _Buffer.Values.Where(q => q.Item2 >= _Access - _MaxCapacity / 2).ToDictionary(z => z.Item1);
                        _Buffer = temp;
                    }
                    return t;                    
                }
            }
        }

        static void Main(string[] args)
        {
            try
            {
                if (args.Length != 3)
                {
                    string xcstr = "usage: TLGCleanDS <input TLG file> <output tlg file> <config file> \n";
                    TLGCleanConfig cfg = new TLGCleanConfig();
                    System.IO.StringWriter sw = new System.IO.StringWriter();
                    new System.Xml.Serialization.XmlSerializer(typeof(TLGCleanConfig)).Serialize(sw, cfg);
                    throw new Exception(xcstr + "\r\nConfig file syntax:\r\n" + sw.ToString() + "\r\n\r\nSelMode = " + SelectionMode.LowerSigma + " or " + SelectionMode.MoreGrains);
                }

                if (System.IO.File.Exists(args[1])) throw new Exception("File " + args[1] + " already exists!");
                if (!System.IO.File.Exists(args[0])) throw new Exception("File " + args[0] + " does not exist!");
                if (!System.IO.File.Exists(args[2]) || !args[2].EndsWith(".xml")) throw new Exception("File " + args[2] + " does not exist or wrong file format!");

                CleanFile(args[0], args[1], args[2]);


            }
            catch (Exception x)
            {
                Console.WriteLine(x.Message);
            }

        } //end Main

        static void CleanFile(string ifile, string ofile, string cfile)
        {
            try
            {
                var threads = new System.Threading.Thread[System.Environment.ProcessorCount];
                SySal.DataStreams.OPERALinkedZone lz = new SySal.DataStreams.OPERALinkedZone(ifile);
                SySal.DataStreams.OPERALinkedZone.Writer wr = null;

                string xmlconfig = ((SySal.OperaDb.ComputingInfrastructure.ProgramSettings)SySal.OperaPersistence.Restore(cfile, typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings))).Settings;
                System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TLGCleanConfig));
                TLGCleanConfig config = (TLGCleanConfig)xmls.Deserialize(new System.IO.StringReader(xmlconfig));

                wr = new SySal.DataStreams.OPERALinkedZone.Writer(ofile, lz.Id, lz.Extents, lz.Center, lz.Transform);
                wr.SetZInfo(lz.Top.TopZ, lz.Top.BottomZ, lz.Bottom.TopZ, lz.Bottom.BottomZ);

                int i, j, tc = 0;
                int xc, yc, ix, iy;
                bool init = false;
                double MinX, MinY, MaxX, MaxY;
                double DPos, DSl;

                var Sequence = new int[] { 0 };

                MaxX = MinX = lz[0].Info.Intercept.X;
                MaxY = MinY = lz[0].Info.Intercept.Y;
                for (i = 0; i < threads.Length; i++)
                {
                    threads[i] = new System.Threading.Thread(() =>
                    {
                        BufferedLinkedZone blz = new BufferedLinkedZone() { _LZ = lz };
                        int ii;
                        while ((ii = System.Threading.Interlocked.Increment(ref Sequence[0])) < lz.Length)
                        {
                            var lzi = blz[ii].Info;
                            if (lzi.Sigma < 0) continue; //skip micro-tracks

                            if (init == false)
                            {
                                MaxX = MinX = lzi.Intercept.X;
                                MaxY = MinY = lzi.Intercept.Y;
                                init = true;
                            }
                            else
                            {
                                if (lzi.Intercept.X < MinX) MinX = lzi.Intercept.X;
                                else if (lzi.Intercept.X > MaxX) MaxX = lzi.Intercept.X;
                                if (lzi.Intercept.Y < MinY) MinY = lzi.Intercept.Y;
                                else if (lzi.Intercept.Y > MaxY) MaxY = lzi.Intercept.Y;
                            }
                        }
                    }); //end loop over base-tracks
                    threads[i].Start();
                }
                foreach (var t in threads) t.Join();

                //divide into cells
                xc = (int)Math.Ceiling((MaxX - MinX) / (config.PosTol * 2));
                yc = (int)Math.Ceiling((MaxY - MinY) / (config.PosTol * 2));

                Console.WriteLine("\n\ntotal nr of tracks: " + lz.Length);
                Console.WriteLine("x=[{0:F1},{1:F1}] \t y=[{2:F1},{3:F1}]", MinX, MaxX, MinY, MaxY);
                Console.WriteLine("dim: {0:F1} \t {1:F1}", Math.Abs(MaxX - MinX), Math.Abs(MaxY - MinY));
                Console.WriteLine("cells: {0} \t {1}", xc, yc);

                int[,] Cells = new int[yc, xc];
                for (iy = 0; iy < yc; iy++)
                    for (ix = 0; ix < xc; ix++)
                        Cells[iy, ix] = 0;

                Console.WriteLine("cell array initialization done!");

                //count tracks in each cell
                Sequence[0] = 0;

                tc = 0;
                for (i = 0; i < threads.Length; i++)
                {
                    threads[i] = new System.Threading.Thread(() =>
                    {
                        BufferedLinkedZone blz = new BufferedLinkedZone() { _LZ = lz };
                        int _tc = 0;
                        int ii;
                        while ((ii = System.Threading.Interlocked.Increment(ref Sequence[0])) < lz.Length)
                        {
                            var lzi = blz[ii].Info;
                            if (lzi.Sigma < 0) //promoted micro-tracks are saved, no selection 
                            {
                                int ti, bi;
                                wr.AddMIPEmulsionTrack(blz[ii].Top.Info, ti = blz[ii].Top.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)blz[ii].Top).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)blz[ii].Top).OriginalRawData, true);
                                wr.AddMIPEmulsionTrack(blz[ii].Bottom.Info, bi = blz[ii].Bottom.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)blz[ii].Bottom).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)blz[ii].Bottom).OriginalRawData, false);
                                wr.AddMIPBasetrack(lzi, tc, ti, bi);
                                _tc++;
                            }
                            else //base-tracks only
                            {
                                int iix = (int)Math.Ceiling((lzi.Intercept.X - MinX) / (config.PosTol * 2)) - 1;
                                int iiy = (int)Math.Ceiling((lzi.Intercept.Y - MinY) / (config.PosTol * 2)) - 1;

                                if (iix > xc) iix = xc;
                                else if (iix < 0) iix = 0;
                                if (iiy > yc) iiy = yc;
                                else if (iiy < 0) iiy = 0;

                                lock (Cells) Cells[iiy, iix]++;
                            }
                        }
                        lock (Cells) tc += _tc;
                    });
                    threads[i].Start();
                };
                foreach (var t in threads) t.Join();
                Console.WriteLine("track counting done!");

                //create track index array for each cell
                int[][] ITracks = new int[xc * yc][];
                for (iy = 0, i = 0; iy < yc; iy++)
                {
                    for (ix = 0; ix < xc; ix++, i++)
                    {
                        ITracks[i] = new int[Cells[iy, ix]]; //for each cell i (iy, ix), allocate memory for Cells[iy, ix] tracks
                        for (j = 0; j < Cells[iy, ix]; j++)
                            ITracks[i][j] = -1;
                    };
                };
                Console.WriteLine("track index array initialization done!");                

                for (i = 0; i < lz.Length; i++)
                {
                    if (lz[i].Info.Sigma > 0)
                    {
                        ix = (int)Math.Ceiling((lz[i].Info.Intercept.X - MinX) / (config.PosTol * 2)) - 1;
                        iy = (int)Math.Ceiling((lz[i].Info.Intercept.Y - MinY) / (config.PosTol * 2)) - 1;

                        if (ix > xc) ix = xc;
                        else if (ix < 0) ix = 0;
                        if (iy > yc) iy = yc;
                        else if (iy < 0) iy = 0;

                        //if (ix > xc || ix < 0 || iy > yc || iy < 0) continue;

                        for (j = 0; j < Cells[iy, ix]; j++)
                        {
                            if (ITracks[xc * iy + ix][j] == -1) //save track index in first available array element 
                            {
                                ITracks[xc * iy + ix][j] = i;
                                j = Cells[iy, ix];
                            };
                        };
                    };
                };

                Console.WriteLine("track cleaning...");

                DateTime startTime = DateTime.Now;
                DateTime startTime2 = startTime;
                int tcount = 0;
                int tcompare = 0;

                //apply track cleaning cell by cell
                for (iy = 0; iy < yc; iy++)
                {
                    //for test only
                    if (iy - (iy / 10) * 10 == 0)
                    {
                        Console.WriteLine("iy=" + iy);
                        Console.WriteLine(" tracks " + tcount + " comparing tracks " + tcompare + 
                            " ratio " + (tcount == 0 ? 0 : tcompare/tcount) + " elapsed time: {0:F1} seconds",(DateTime.Now - startTime2).TotalSeconds);
                        startTime2 = DateTime.Now;
                        tcount = 0;
                        tcompare = 0;
                    };                    

                    for (ix = 0; ix < xc; ix++)
                    {
                        //for test only
                        tcount += Cells[iy, ix];

                        Sequence[0] = 0;
                        for (i = 0; i < threads.Length; i++)
                        {
                            threads[i] = new System.Threading.Thread(() =>
                            {
                                int ii;
                                BufferedLinkedZone blz = new BufferedLinkedZone() { _LZ = lz };
                                while ((ii = System.Threading.Interlocked.Increment(ref Sequence[0])) < Cells[iy, ix])
                                {
                                    int iji = ITracks[xc * iy + ix][ii];
                                    int ijj;
                                    if (iji == -1) return; //skip 
                                    var tk = blz[iji];
                                    int _tcompare = 0;
                                    int jj, k1, k2;
                                    //...loop over current cell and neighbouring cells
                                    for (k1 = (iy - 1 > 0) ? (iy - 1) : iy; k1 <= (iy + 1) && k1 < yc; k1++)
                                    {
                                        for (k2 = (ix - 1 > 0) ? (ix - 1) : ix; k2 <= (ix + 1) && k2 < xc; k2++)
                                        {
                                            for (jj = 0; jj < Cells[k1, k2]; jj++) //...and compare with other tracks
                                            {
                                                _tcompare++;
                                                if ((k1 == iy && k2 == ix && jj == i) || (ijj = ITracks[xc * k1 + k2][jj]) == -1) continue; //skip if current track or alreedy deleted track 
                                                {
                                                    var tkj = blz[ijj];
                                                    DPos = Math.Sqrt((tkj.Info.Intercept.X - tk.Info.Intercept.X) * (tkj.Info.Intercept.X - tk.Info.Intercept.X) + (tkj.Info.Intercept.Y - tk.Info.Intercept.Y) * (tkj.Info.Intercept.Y - tk.Info.Intercept.Y));
                                                    DSl = Math.Sqrt((tkj.Info.Slope.X - tk.Info.Slope.X) * (tkj.Info.Slope.X - tk.Info.Slope.X) + (tkj.Info.Slope.Y - tk.Info.Slope.Y) * (tkj.Info.Slope.Y - tk.Info.Slope.Y)); ;
                                                    if (DPos < config.PosTol && DSl < config.SlopeTol)
                                                    {
                                                        //...then select according to the following criterion
                                                        if ((config.SelMode == SelectionMode.LowerSigma && (tk.Info.Sigma < tkj.Info.Sigma)) || (config.SelMode == SelectionMode.MoreGrains && (tk.Info.Count > tkj.Info.Count)))
                                                        {
                                                            ITracks[xc * k1 + k2][jj] = -1;
                                                        }
                                                        else
                                                        {
                                                            ITracks[xc * iy + ix][ii] = -1;                                                            
                                                        };
                                                    };
                                                };
                                            }; //end loop j                                        
                                        }; //end loop k2
                                    }; //end loop k1
                                    lock (Cells) tcompare += _tcompare;
                                }
                            });
                            threads[i].Start();
                        }; //end loop i
                        foreach (var t in threads) t.Join();

                    }; //end loop ix 

                }; //end loop iy
                Console.WriteLine(" total elapsed time for track cleaning: {0:F1} seconds", (DateTime.Now - startTime).TotalSeconds);

                Console.WriteLine("saving surviving tracks to file...");
                //save surviving base-tracks to file 
                for (iy = 0, i = 0; iy < yc; iy++)
                {
                    for (ix = 0; ix < xc; ix++, i++)
                    {
                        for (i = 0; i < Cells[iy, ix]; i++)
                        {
                            if (ITracks[xc * iy + ix][i] != -1)
                            {
                                int ti, bi;
                                wr.AddMIPEmulsionTrack(lz[ITracks[xc * iy + ix][i]].Top.Info, ti = lz[ITracks[xc * iy + ix][i]].Top.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[ITracks[xc * iy + ix][i]].Top).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[ITracks[xc * iy + ix][i]].Top).OriginalRawData, true);
                                wr.AddMIPEmulsionTrack(lz[ITracks[xc * iy + ix][i]].Bottom.Info, bi = lz[ITracks[xc * iy + ix][i]].Bottom.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[ITracks[xc * iy + ix][i]].Bottom).View.Id, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[ITracks[xc * iy + ix][i]].Bottom).OriginalRawData, false);
                                wr.AddMIPBasetrack(lz[ITracks[xc * iy + ix][i]].Info, tc, ti, bi);
                                tc++;
                            };
                        };
                    };
                };
                if (wr != null)
                {
                    for (i = 0; i < ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Top).ViewCount; i++)
                        wr.AddView(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Top).View(i), true);
                    for (i = 0; i < ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Bottom).ViewCount; i++)
                        wr.AddView(((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)lz.Bottom).View(i), false);
                }
                if (wr != null) wr.Complete();

                Console.WriteLine("total nr of tracks: " + lz.Length);
                Console.WriteLine("nr of surviving tracks: " + tc);

        }
        catch (Exception x)
        {
            Console.WriteLine(x.Message);
        }

        } //end Clean


    } // end class Exe
}
