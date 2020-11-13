using System;
using System.Web;
using System.Net;
using System.Collections.Generic;

namespace SySal.Executables.PostProcessingManager
{
    partial class Program
    {
        static System.Text.RegularExpressions.Regex rx_JobCreated = new System.Text.RegularExpressions.Regex(@"CREATED\s+(\S+)");

        public static string HTTPScheduleJob(string hostname, int qn, string startfilename, int stripid)
        {
            using (System.Net.WebClient cl = new System.Net.WebClient())
            {
                int trials;
                for (trials = 3; trials >= 0; trials--)
                {
                    System.IO.Stream rstr = null;
                    try
                    {
                        rstr = cl.OpenRead("http://" + hostname + ":1783/addrwdj?queue=" + qn + "&rwdz=" +
                            System.Web.HttpUtility.UrlEncode(startfilename) +
                            "&rwds=" + stripid +
                            "&rwdd=" + System.Web.HttpUtility.UrlEncode(C.OutputDir)
                            );
                        int b;
                        List<byte> bytes = new List<byte>();
                        while ((b = rstr.ReadByte()) >= 0)
                            bytes.Add((byte)b);
                        string result = System.Text.Encoding.UTF8.GetString(bytes.ToArray()).Trim();
                        System.Text.RegularExpressions.Match m = rx_JobCreated.Match(result);
                        if (m.Success)
                            return m.Groups[1].Value;
                        System.Threading.Thread.Sleep((4 - trials) * 1000);
                    }
                    catch (Exception x)
                    {
                        if (C.EnableLog)
                            Program.Log("Error scheduling job :\"" + x.Message + "\" on " + hostname + ".");
                        return null;
                    }
                    finally
                    {
                        if (rstr != null) rstr.Close();
                    }
                }
            }
            return null;
        }

        public static string HTTPScheduleJob(string hostname, int qn, string startfilename, int stripid, int firstview, int lastview, int fragmentindex)
        {
            using (System.Net.WebClient cl = new System.Net.WebClient())
            {
                int trials;
                for (trials = 3; trials >= 0; trials--)
                {
                    System.IO.Stream rstr = null;
                    try
                    {
                        rstr = cl.OpenRead("http://" + hostname + ":1783/addrwdj?queue=" + qn + "&rwdz=" +
                            System.Web.HttpUtility.UrlEncode(startfilename) +
                            "&rwds=" + stripid + "&rwdfv=" + firstview + "&rwdlv=" + lastview + "&rwdi=" + fragmentindex +
                            "&rwdd=" + System.Web.HttpUtility.UrlEncode(C.OutputDir)
                            );
                        int b;
                        List<byte> bytes = new List<byte>();
                        while ((b = rstr.ReadByte()) >= 0)
                            bytes.Add((byte)b);
                        string result = System.Text.Encoding.UTF8.GetString(bytes.ToArray()).Trim();
                        System.Text.RegularExpressions.Match m = rx_JobCreated.Match(result);
                        if (m.Success)
                            return m.Groups[1].Value;
                        System.Threading.Thread.Sleep((4 - trials) * 1000);
                    }
                    catch (Exception x)
                    {
                        if (C.EnableLog)
                            Program.Log("Error scheduling job :\"" + x.Message + "\" on " + hostname + ".");
                        return null;
                    }
                    finally
                    {
                        if (rstr != null) rstr.Close();
                    }
                }
            }
            return null;
        }

        public enum JobOutcome { Failed, Done, Waiting, Unknown };

        public static JobOutcome HTTPCheckJob(string hostname, int qn, string jobname)
        {
            using (System.Net.WebClient cl = new System.Net.WebClient())
            {
                for (int trials = 3; trials >= 0; trials--)
                {
                    System.IO.Stream rstr = null;
                    try
                    {
                        rstr = cl.OpenRead("http://" + hostname + ":1783/queryjob?queue=" + qn + "&j=" + jobname);
                        int b;
                        List<byte> bytes = new List<byte>();
                        while ((b = rstr.ReadByte()) >= 0)
                            bytes.Add((byte)b);
                        string result = System.Text.Encoding.UTF8.GetString(bytes.ToArray()).Trim();
                        switch (result)
                        {
                            case "FAILED": return JobOutcome.Failed; 

                            case "DONE": return JobOutcome.Done; 

                            case "UNKNOWN": return JobOutcome.Unknown; 

                            default: return JobOutcome.Waiting;
                        }
                    }
                    catch (Exception x)
                    {
                        if (C.EnableLog)
                            Program.Log("Error checking job :\"" + x.Message + "\" on " + hostname + ".");
                        System.Threading.Thread.Sleep((4 - trials) * 1000);
                    }
                    finally
                    {
                        if (rstr != null) rstr.Close();
                    }
                }
                return JobOutcome.Unknown;
            }
        }

        public static int HTTPCheckTrackerNumber(string hostname)
        {
            int trials;
            using (System.Net.WebClient cl = new System.Net.WebClient())
            {
                for (trials = 3; trials >= 0; trials--)
                {
                    System.IO.Stream rstr = null;
                    try
                    {
                        rstr = cl.OpenRead("http://" + hostname + ":1783/qn");
                        int b;
                        List<byte> bytes = new List<byte>();
                        while ((b = rstr.ReadByte()) >= 0)
                            bytes.Add((byte)b);
                        return int.Parse(System.Text.Encoding.UTF8.GetString(bytes.ToArray()));
                    }
                    catch (Exception)
                    {
                        System.Threading.Thread.Sleep((4 - trials) * 1000);
                    }
                    finally
                    {
                        if (rstr != null) rstr.Close();
                    }
                }
            }
            return -1;
        }
    }
}