using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Services;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;

namespace SySal.Executable.ListScan
{
    class Program
    {
        const string StrLoadSetup = "loadsetup";
        const string StrLoadPlate = "loadplate";
        const string StrScan = "scan";
        const string StrRun = "run";
        const string StrBatch = "batch";

        static System.Text.RegularExpressions.Regex rx_linenum = new System.Text.RegularExpressions.Regex(@"\s*([-0123456789]+)\s+(" + StrLoadSetup + "|" + StrLoadPlate + "|" + StrScan + "|" + StrRun + "|" + StrBatch + @")\s+");

        abstract class Command : IComparable
        {
            public int LineNumber;
            public string Text;

            public Command(int linenum, string txt) { LineNumber = linenum; Text = txt; }
            abstract public bool Execute();

            #region IComparable Members

            public int CompareTo(object obj)
            {
                return Math.Abs(LineNumber) - Math.Abs(((Command)obj).LineNumber);
            }

            #endregion
        }

        class CmdLoadSetup : Command
        {
            public string File;

            public CmdLoadSetup(int linenum, string txt, int firstindex)
                : base(linenum, txt)
            {
                File = txt.Substring(firstindex).Trim();
            }

            public override string ToString()
            {
                return StrLoadSetup + " " + File;
            }

            public override bool Execute()
            {
                Console.WriteLine();
                Console.Write(LineNumber + "->" + StrLoadSetup + "->");
                bool setscanlayout = ScanSrv.SetScanLayout(System.IO.File.ReadAllText(File));
                Console.Write(setscanlayout);
                return setscanlayout;
            }

            public static string Explain { get { return StrLoadSetup + " <file>"; } }
        }

        class CmdLoadPlate : Command
        {
            public uint Brick;
            public uint Plate;
            public string Desc;
            public string Marks;

            static System.Text.RegularExpressions.Regex rx_args = new System.Text.RegularExpressions.Regex(@"(\d+)\s*,\s*(\d+)\s*,([^,]*),([^,]*)");

            public CmdLoadPlate(int linenum, string txt, int firstindex)
                : base(linenum, txt)
            {
                System.Text.RegularExpressions.Match m = rx_args.Match(txt, firstindex);
                if (m.Success == false || (m.Index + m.Length) != txt.Length) throw new Exception("Syntax error.");
                Brick = Convert.ToUInt32(m.Groups[1].Value);
                Plate = Convert.ToUInt32(m.Groups[2].Value);
                Desc = m.Groups[3].Value;
                Marks = m.Groups[4].Value;
            }

            public override string ToString()
            {
                return StrLoadPlate + " " + Brick + "," + Plate + "," + Desc + "," + Marks;
            }

            public override bool Execute()
            {
                Console.WriteLine();
                Console.Write(LineNumber + "->" + StrLoadPlate + "->");
                SySal.DAQSystem.Scanning.MountPlateDesc desc = new SySal.DAQSystem.Scanning.MountPlateDesc();
                desc.BrickId = Brick;
                desc.PlateId = Plate;
                desc.TextDesc = Desc;
                desc.MapInitString = Marks;
                bool load = ScanSrv.LoadPlate(desc);
                Console.Write(load);
                return load;
            }

            public static string Explain { get { return StrLoadPlate + " <brickid>,<plateid>,<text>,<marks>"; } }
        }

        class CmdScan : Command
        {
            public long Id;
            public double MinX, MaxX, MinY, MaxY;
            public bool UseSlopePresel;
            public double MinSX, MaxSX, MinSY, MaxSY;
            public string OutputFile;

            static System.Text.RegularExpressions.Regex rx_args = new System.Text.RegularExpressions.Regex(@"(\d+)\s*,([^,]*),([^,]*),([^,]*),([^,]*),\s*(true|false)\s*,([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)");

            public CmdScan(int linenum, string txt, int firstindex)
                : base(linenum, txt)
            {
                System.Text.RegularExpressions.Match m = rx_args.Match(txt, firstindex);
                if (m.Success == false || (m.Index + m.Length) != txt.Length) throw new Exception("Syntax error.");
                Id = Convert.ToInt64(m.Groups[1].Value);
                MinX = Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                MaxX = Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                MinY = Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                MaxY = Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                UseSlopePresel = Convert.ToBoolean(m.Groups[6].Value);
                MinSX = Convert.ToDouble(m.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture);
                MaxSX = Convert.ToDouble(m.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture);
                MinSY = Convert.ToDouble(m.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                MaxSY = Convert.ToDouble(m.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                OutputFile = m.Groups[11].Value.Trim();
            }

            public override string ToString()
            {
                return StrScan + " " + Id +
                    "," + MinX.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + MaxX.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + MinY.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + MaxY.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + (UseSlopePresel ? "true" : "false") +
                    "," + MinSX.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + MaxSX.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + MinSY.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + MaxSY.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    "," + OutputFile;
            }

            public override bool Execute()
            {
                Console.WriteLine();
                Console.Write(LineNumber + "->" + StrScan + "->");
                SySal.DAQSystem.Scanning.ZoneDesc zd = new SySal.DAQSystem.Scanning.ZoneDesc();
                zd.Series = Id;
                zd.MinX = MinX;
                zd.MinY = MinY;
                zd.MaxX = MaxX;
                zd.MaxY = MaxY;
                zd.UsePresetSlope = UseSlopePresel;
                zd.PresetSlope.X = 0.5 * (MinSX + MaxSX);
                zd.PresetSlope.Y = 0.5 * (MinSY + MaxSY);
                zd.PresetSlopeAcc.X = 0.5 * (MaxSX - MinSX);
                zd.PresetSlopeAcc.Y = 0.5 * (MaxSY - MinSY);
                zd.Outname = OutputFile;
                bool scan = ScanSrv.Scan(zd);
                Console.Write(scan);
                return scan;
            }

            public static string Explain { get { return StrScan + " <id>,<minx>,<maxx>,<miny>,<maxy>,<useslopesel>,<minsx>,<maxsx>,<minsy>,<maxsy>,<outputfile>"; } }
        }

        class CmdRun : Command
        {
            public string ExeFile;
            public string[] Arguments;

            public CmdRun(int linenum, string txt, int firstindex)
                : base(linenum, txt)
            {
                string[] tokens = txt.Substring(firstindex).Split(',');
                ExeFile = tokens[0].Trim();
                Arguments = new string[tokens.Length - 1];
                int i;
                for (i = 0; i < Arguments.Length; i++)
                    Arguments[i] = tokens[i + 1].Trim();
            }

            public override string ToString()
            {
                string str = StrRun + " " + ExeFile;
                foreach (string s in Arguments)
                    str += "," + s;
                return str;
            }

            public override bool Execute()
            {
                string s = "";
                int i;
                for (i = 0; i < Arguments.Length; i++)
                {
                    if (i > 0) s += " ";
                    if (Arguments[i].StartsWith("\"") == false) s += "\"" + Arguments[i] + "\"";
                    else s += Arguments[i];
                }
                System.Diagnostics.Process.Start(ExeFile, s);
                return true;
            }

            public static string Explain { get { return StrRun + " <exefile>,<arg1>,<arg2>,<arg3>..."; } }
        }

        class CmdBatch : Command
        {
            public string BatchServer;
            public ulong BatchId;
            public uint MachinePowerClass;
            public string LogFile;
            public string ExeFile;
            public string[] Arguments;

            public CmdBatch(int linenum, string txt, int firstindex)
                : base(linenum, txt)
            {
                string[] tokens = txt.Substring(firstindex).Split(',');
                BatchServer = tokens[0].Trim();
                BatchId = 0;
                MachinePowerClass = Convert.ToUInt32(tokens[1]);
                LogFile = tokens[2].Trim();
                ExeFile = tokens[3].Trim();
                Arguments = new string[tokens.Length - 4];
                int i;
                for (i = 0; i < Arguments.Length; i++)
                    Arguments[i] = tokens[i + 4].Trim();
            }

            public override string ToString()
            {
                string str = StrBatch + " " + BatchServer + "," + MachinePowerClass + "," + LogFile + "," + ExeFile;
                foreach (string s in Arguments)
                    str += "," + s;
                return str;
            }

            public override bool Execute()
            {
                SySal.DAQSystem.IDataProcessingServer DataProcSrv = (SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + BatchServer + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer) + "/DataProcessingServer.rem");
                SySal.DAQSystem.DataProcessingBatchDesc bd = new SySal.DAQSystem.DataProcessingBatchDesc();
                SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
                bd.Username = cred.OPERAUserName;
                bd.Password = cred.OPERAPassword;
                bd.Description = "ListScan - " + LineNumber;
                bd.Id = DataProcSrv.SuggestId;
                bd.MachinePowerClass = MachinePowerClass;
                bd.OutputTextSaveFile = LogFile;
                bd.Filename = ExeFile;
                bd.AliasPassword = "";
                bd.AliasUsername = "";
                Console.Write("(batch=" + bd.Id.ToString("X16") + ")");
                string s = "";
                int i;
                for (i = 0; i < Arguments.Length; i++)
                {
                    if (i > 0) s += " ";
                    if (Arguments[i].StartsWith("\"") == false) s += "\"" + Arguments[i] + "\"";
                    else s += Arguments[i];
                }
                bd.CommandLineArguments = s;
                return DataProcSrv.Enqueue(bd);
            }

            public static string Explain { get { return StrBatch + " <batchsrv>,<powerclass>,<logfile>,<exefile>,<arg1>,<arg2>,<arg3>..."; } }
        }

        static SySal.DAQSystem.ScanServer ScanSrv;

        static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("Usage: ListScanV5 <ip> <listfile>");
                Console.WriteLine("Each command must begin with a number, and commands are executed in increasing order.");
                Console.WriteLine("Executed commands are rewritten to the file with a negative number.");
                Console.WriteLine("<taskid> " + CmdLoadSetup.Explain);
                Console.WriteLine("<taskid> " + CmdLoadPlate.Explain);
                Console.WriteLine("<taskid> " + CmdScan.Explain);
                Console.WriteLine("<taskid> " + CmdRun.Explain);
                Console.WriteLine("<taskid> " + CmdBatch.Explain);
                return;
            }
            ScanSrv = (SySal.DAQSystem.ScanServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.ScanServer), "tcp://" + args[0] + ":" + ((int)SySal.DAQSystem.OperaPort.ScanServer) + "/ScanServer.rem");
            string[] lines = System.IO.File.ReadAllText(args[1]).Split('\n', '\r');
            System.Collections.ArrayList cmd = new System.Collections.ArrayList();
            int errors = 0;
            foreach (string line in lines)
                if (line.Trim().Length > 0)
                {
                    System.Text.RegularExpressions.Match m = rx_linenum.Match(line);
                    if (m.Success == false)
                    {
                        errors++;
                        Console.WriteLine("Cannot understand line \"" + line + "\"");
                    }
                    else
                        try
                        {
                            int linenum = Convert.ToInt32(m.Groups[1].Value);
                            switch (m.Groups[2].Value)
                            {
                                case StrLoadSetup: cmd.Add(new CmdLoadSetup(linenum, line, m.Index + m.Length)); break;

                                case StrLoadPlate: cmd.Add(new CmdLoadPlate(linenum, line, m.Index + m.Length)); break;

                                case StrScan: cmd.Add(new CmdScan(linenum, line, m.Index + m.Length)); break;

                                case StrRun: cmd.Add(new CmdRun(linenum, line, m.Index + m.Length)); break;

                                case StrBatch: cmd.Add(new CmdBatch(linenum, line, m.Index + m.Length)); break;

                                default: Console.WriteLine("Unknown command \"" + m.Groups[2].Value + "\"."); errors++; break;
                            }
                        }
                        catch (Exception x)
                        {
                            Console.WriteLine("Error in line \"" + line + "\": " + x.ToString());
                            errors++;
                        }
                }
            if (errors > 0)
            {
                Console.Error.WriteLine(errors + " error(s) found; aborting.");
                return;
            }
            Console.WriteLine(cmd.Count + " Commands read.");
            int cmdexec = 0;
            do
            {
                cmd.Sort();
                string rewfile = "";
                foreach (Command c in cmd)
                    rewfile += "\r\n" + c.LineNumber + " " + c.ToString();
                try
                {
                    System.IO.File.WriteAllText(args[1], rewfile);
                }
                catch (Exception x)
                {
                    Console.Error.WriteLine("Cannot update list file!\r\nAborting\r\n" + x.ToString());
                    return;
                }
                int ln;
                for (cmdexec = 0; cmdexec < cmd.Count; cmdexec++)
                    if ((ln = (cmd[cmdexec] as Command).LineNumber) > 0)
                    {
                        (cmd[cmdexec] as Command).Execute();
                        (cmd[cmdexec] as Command).LineNumber = -ln;
                        break;
                    }
            }
            while (cmdexec != cmd.Count);
        }
    }
}
