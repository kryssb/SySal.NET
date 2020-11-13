using System;
using System.Linq;

namespace SySal.Executables.PostProcessingManager
{
    class CodeTest
    {
        public const string TestServer = "test";

        internal static void TestJobSplit(string zdir)
        {
#if ENABLE_TESTS                        
            Console.WriteLine("Test Job splitting");
            try
            {
                var zone = System.IO.Directory.GetFiles(zdir, "*.scan").Single();
                var z = SySal.Executables.PostProcessingManager.Zone.FromFile(zone);
                z.OutDir = zdir;                                
                z.AutoDelete = false;
                Console.WriteLine("Read zone " + z.ToString());
                uint i;
                for (i = 0; i < z.Strips; i++)
                {
                    Console.WriteLine("Making job " + i);
                    SySal.Executables.PostProcessingManager.Job j = new Executables.PostProcessingManager.Job() { OwnerZone = z, FirstView = 0, LastView = z.Views - 1, JobName = "testjob", StripId = i };
                    Console.WriteLine("Splitting job " + j.ToString());
                    var jobs = j.TrySplitJob(Math.Min(i, z.Views * 2));
                    foreach (var j1 in jobs)
                        Console.WriteLine("Obtained " + j1.ToString());
                }
                Console.WriteLine("Dumping catalog with " + z.GetRWC().Fragments + " fragments");
                var fi = z.GetRWC().GetFragmentIndices();
                for (uint iy = 0; iy < z.Strips; iy++)
                {
                    for (uint ix = 0; ix < z.Views; ix++)
                        Console.Write(fi[iy, ix] + " ");
                    Console.WriteLine();
                }
            }
            catch (Exception x)
            {
                Console.WriteLine(x.ToString());
            }
            Console.WriteLine("End test");
#endif
        }

        internal static void TestAddRemoveTracker(string zdir)
        {
#if ENABLE_TESTS
            var zone = System.IO.Directory.GetFiles(zdir, "*.scan").Single();
            var z = SySal.Executables.PostProcessingManager.Zone.FromFile(zone);
            z.OutDir = zdir;
            z.AutoDelete = false;
            Console.WriteLine("Test Add/Remove Tracker");
            Console.WriteLine("Removing nonexisting tracker");
            Program.AddRemoveTracker(TestServer, false);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Adding tracker");
            Program.AddRemoveTracker(TestServer, true);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Adding queues");
            var srv = Program.UpdateTrackerQueuesAndGet(TestServer, HTTPCheckTrackerNumber(TestServer));
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Adding job");
            var sp = new PostProcessingManager.Job() { OwnerZone = z, Completed = false, FirstView = 0, LastView = z.Views - 1, JobName = "testjob", Split = false, StripId = z.Strips / 2 };
            sp.ForceFragmentIndex(sp.StripId + 1);
            Program.ScheduleJob(TestServer, srv, 0, sp);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Adding duplicate tracker");
            Program.AddRemoveTracker(TestServer, true);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Removing tracker");
            Program.AddRemoveTracker(TestServer, false);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Removing nonexisting tracker");
            Program.AddRemoveTracker(TestServer, false);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Adding tracker");
            Program.AddRemoveTracker(TestServer, true);
            srv = Program.UpdateTrackerQueuesAndGet(TestServer, HTTPCheckTrackerNumber(TestServer));
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Scheduling job");
            Program.ScheduleFirstActiveJob(TestServer, srv);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            lock (_CurrentJob)
                for (int i = 0; i < _CurrentJob.Length; i++)
                    if (_CurrentJob[i] != null)
                        _CurrentJob[i] = FailurePrefix + _CurrentJob[i];
            Console.WriteLine("Failing job");
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Program.ScheduleFirstActiveJob(TestServer, srv);
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Succeeding job");
            lock (_CurrentJob)
                for (int i = 0; i < _CurrentJob.Length; i++)
                    if (_CurrentJob[i] != null)
                        _CurrentJob[i] = SuccessPrefix + _CurrentJob[i];
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Program.AddRemoveTracker(TestServer, false);
            lock (_CurrentJob)
                for (int i = 0; i < _CurrentJob.Length; i++)
                    _CurrentJob[i] = null;
            foreach (var rwd in System.IO.Directory.GetFiles(zdir, "*.rwd.????????"))
                System.IO.File.Delete(rwd);
            Console.WriteLine("End test");
#endif
        }

        internal static void TestProcessZone(string zone)
        {
#if ENABLE_TESTS
            Console.WriteLine("Test process zone");
            Console.WriteLine("Adding tracker");
            Program.AddRemoveTracker(TestServer, true);
            Console.WriteLine("Getting tracker info");
            var srv = Program.UpdateTrackerQueuesAndGet(TestServer, HTTPCheckTrackerNumber(TestServer));
            Program.supv_Elapsed(null, null);
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            Console.WriteLine("Creating zone");
            Program.TestCreatedWithOptions(zone, false);
            Console.WriteLine("Getting RWC info");
            var js = Program.GetJobs();
            Console.WriteLine("Jobs: " + js.Count());
            var z = Program.GetJobs().Select(j => j.OwnerZone).Distinct().Single();
            var rwc = z.GetRWC();            
            foreach (var rwd in System.IO.Directory.GetFiles(z.OutDir, "*.rwd.????????"))
                System.IO.File.Delete(rwd);
            var fragindices = rwc.GetFragmentIndices();
            Console.WriteLine("Executing jobs");
            Program.supv_Elapsed(null, null);
            z.Update();
            Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
            int totaljobs = 0;
            while (Program.GetJobs().Count() > 0)
            {
                z.Update();                
                Program.ScheduleFirstActiveJob(TestServer, srv);
                Console.WriteLine("Succeeding job");
                lock (_CurrentJob)
                    for (int i = 0; i < _CurrentJob.Length; i++)
                        if (_CurrentJob[i] != null)
                        {
                            totaljobs++;
                            _CurrentJob[i] = SuccessPrefix + _CurrentJob[i];
                        }
                Program.supv_Elapsed(null, null);
                Console.WriteLine("Monitoring Info: " + Program.MonInfo.ToString());
                Console.WriteLine("Fragments " + z.GetRWC().Fragments);
            }
            Console.WriteLine("Loading RWC");
            using (var fs = new System.IO.FileStream(z.GetRWCFileName(z.OutDir), System.IO.FileMode.Open, System.IO.FileAccess.Read))
            {
                var rwcsaved = new SySal.Scanning.Plate.IO.OPERA.RawData.Catalog(fs);
                Console.WriteLine("Fragments " + rwcsaved.Fragments + " Jobs " + totaljobs);
                int errfound = 0;
                for (int iy = 0; iy < rwcsaved.YSize; iy++)
                {
                    Console.WriteLine();
                    for (int ix = 0; ix < rwcsaved.XSize; ix++)
                    {
                        var f = rwcsaved[iy, ix];
                        Console.Write(f + " ");
                        if (f < 1 || f > rwcsaved.Fragments)
                        {
                            errfound++;
                            Console.WriteLine("Found fragment index " + f + " at iy = " + iy + ", ix = " + ix + ".");
                        }
                    }
                }
                if (rwcsaved.Fragments != totaljobs)
                    throw new Exception("Jobs and fragments do not match!");
                if (errfound > 0)
                {
                    throw new Exception("Found " + errfound + " errors.");
                }
            }
            foreach (var rwd in System.IO.Directory.GetFiles(z.OutDir, "*.rwd.????????"))
                System.IO.File.Delete(rwd);
            Console.WriteLine();
            Program.AddRemoveTracker(TestServer, false);
            for (int i = 0; i < _CurrentJob.Length; i++)
                _CurrentJob[i] = null;
            Console.WriteLine("End test");
#endif
        }

        static string [] _CurrentJob = new string[2];

        public static string HTTPScheduleJob(string hostname, int qn, string filetemplate, int stripid, int firstview, int lastview, int fragmentindex)
        {
#if ENABLE_TESTS            
            if (qn < 0 || qn >= 2) return null;            
            lock (_CurrentJob)
            {
                if (_CurrentJob[qn] != null)
                {
                    Console.WriteLine("Worker " + qn + " busy, refusing job");
                    return null;
                }
                var s = filetemplate + ".twr";
                Console.WriteLine("Writing \"" + s + "\" (" + stripid + ", " + firstview + ", " + lastview + ", " + fragmentindex + ") - (worker " + qn + ").");
                System.IO.File.WriteAllText(s, "Strip " + stripid + " firstview " + firstview + " lastview " + lastview + " fragmentindex " + fragmentindex);
                _CurrentJob[qn] = System.Guid.NewGuid().ToString();
                Console.WriteLine("Worker " + qn + " generated job name " + _CurrentJob[qn]);
                return _CurrentJob[qn];
            }
#endif
        }

        public static string HTTPScheduleJob(string hostname, int qn, string rwdfilename, int stripid)
        {
#if ENABLE_TESTS            
            if (qn < 0 || qn >= 2) return null;
            lock (_CurrentJob)
            { 
                if (_CurrentJob[qn] != null)
                {
                    Console.WriteLine("Worker " + qn + " busy, refusing job");
                    return null;
                }                       
                var s = rwdfilename + ".twr";
                Console.WriteLine("Writing \"" + s + "\" (worker " + qn + ").");
                System.IO.File.WriteAllText(s, "Strip " + stripid);
                _CurrentJob[qn] = System.Guid.NewGuid().ToString();
                Console.WriteLine("Worker " + qn + " generated job name " + _CurrentJob[qn]);
                return _CurrentJob[qn];
            }
#endif
        }

        const string SuccessPrefix = " ";
        const string FailurePrefix = "~";

        public static Program.JobOutcome HTTPCheckJob(string hostname, int qn, string jobname)
        {
#if ENABLE_TESTS    
            if (qn < 0 || qn >= 2) return Program.JobOutcome.Unknown;
            lock(_CurrentJob)
            {
                if (_CurrentJob[qn] == null) return Program.JobOutcome.Unknown;
                if (string.Compare(_CurrentJob[qn], jobname, true) == 0) return Program.JobOutcome.Waiting;
                if (string.Compare(_CurrentJob[qn], SuccessPrefix + jobname, true) == 0)
                {
                    _CurrentJob[qn] = null;                    
                    return Program.JobOutcome.Done;
                }
                if (string.Compare(_CurrentJob[qn], FailurePrefix + jobname, true) == 0)
                {
                    _CurrentJob[qn] = null;
                    return Program.JobOutcome.Failed;
                }
                return Program.JobOutcome.Unknown;
            }
#endif
        }

        public static int HTTPCheckTrackerNumber(string hostname)
        {
#if ENABLE_TESTS
            return _CurrentJob.Length;
#endif
        }
    }
}