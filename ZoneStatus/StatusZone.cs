using System;
using System.Collections;
using System.IO;
using System.Runtime.Serialization;
using System.Xml;
using System.Xml.Serialization;
using NumericalTools;
using SySal.Scanning.Plate.IO.OPERA.RawData;
using SySal.Scanning.Plate.IO.OPERA;

namespace ZoneStatus
{
	class MIPIndexedEmulsionTrack : SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack
	{
		public MIPIndexedEmulsionTrack(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack t, int id)
		{
			m_Info = MIPIndexedEmulsionTrack.AccessInfo(t);
			m_Grains = MIPIndexedEmulsionTrack.AccessGrains(t);
			m_Id = id;
			m_OriginalRawData = t.OriginalRawData;
		}
	}
	/// <summary>
	/// Quality check for a fragment.
	/// </summary>
	[Serializable]
	public class FragmentCheck
	{
		public int Index;
		public double Density;

		public FragmentCheck()
		{
			Index = -1;
			Density = -1;
		}

		public FragmentCheck(FragmentCheck check)
		{
			Index = check.Index;
			Density = check.Density;
		}
	}

	/// <summary>
	/// Quality check for a strip.
	/// </summary>
	[Serializable]
	public class StatusZone 
	{
		public int Attempts;
		public double XTopShrink;
		public double YTopShrink;
		public double XBotShrink;
		public double YBotShrink;

		//internal static uint XFrag, YFrag;

		public FragmentCheck [] FragCheck;

		internal static XmlSerializer ser = new XmlSerializer(typeof(StatusZone));

		public StatusZone()
		{
			Attempts = 0;
			XTopShrink = .0;
			YTopShrink = .0;
			XBotShrink = .0;
			YBotShrink = .0;

			FragCheck = null;
		}

		public StatusZone(StatusZone info)
		{
			Attempts = info.Attempts;
			XTopShrink = info.XTopShrink;
			YTopShrink = info.YTopShrink;
			XBotShrink = info.XBotShrink;
			YBotShrink = info.YBotShrink;
			FragCheck = new FragmentCheck[info.FragCheck.Length];
			int i;
			for (i=0; i<FragCheck.Length; i++) FragCheck[i] = info.FragCheck[i];
		}

		public void Copy(StatusZone info)
		{
			this.Attempts = info.Attempts;
			this.XTopShrink = info.XTopShrink;
			this.YTopShrink = info.YTopShrink;
			this.XBotShrink = info.XBotShrink;
			this.YBotShrink = info.YBotShrink;
			if(info.FragCheck != null) 
			{
				this.FragCheck = new FragmentCheck[info.FragCheck.Length];
				int i;
				for (i=0; i<FragCheck.Length; i++) this.FragCheck[i] = info.FragCheck[i];
			}
			else FragCheck = null;
			return;
		}

		public FragmentCheck this [int index]
		{
			get
			{
				return FragCheck[index];
			}
			set 
			{
				FragCheck[index] = value;
			}
		}

		public void MakeFragmentCheckArray(uint n)
		{
			FragCheck = new FragmentCheck[n];
		}

		private void Write(string filename)
		{
			FileStream fs = new FileStream(filename, FileMode.Create);
			try 
			{
				ser.Serialize(fs, this);
			}
			catch (SerializationException e) 
			{
				Console.WriteLine("Failed to serialize. Reason: " + e.Message);
				throw;
			}
			finally 
			{
				fs.Close();
			}
		}

		public void Read(string filename)
		{
			StatusZone info = null;
			FileStream fs = new FileStream(filename, FileMode.Open);
			try 
			{
				info = (StatusZone) ser.Deserialize(fs);
			}
			catch (SerializationException e) 
			{
				Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
				throw;
			}
			finally 
			{
				this.Copy(info);
				fs.Close();
			}
			return;
		}

		private ComputationResult ComputeShrinkage(LinkedZone lz, bool isTop)
		{
			int NData = lz.Length;
			double [] x = new double[NData];
			double [] dx = new double[NData];
			double [] y = new double[NData];
			double [] dy = new double[NData];

			int i;
			for(i = 0; i < NData; i++)
			{
				if(isTop) x[i] = lz.Top[i].Info.Slope.X;
				else x[i] = lz.Bottom[i].Info.Slope.X;
				dx[i] = lz[i].Info.Slope.X - x[i];
				if(isTop) y[i] = lz.Top[i].Info.Slope.Y;
				else y[i] = lz.Bottom[i].Info.Slope.Y;
				dy[i] = lz[i].Info.Slope.Y - y[i];
			}

			double a=0, b=0, range=0, erry=0, erra=0, errb=0, ccor=0;

			NumericalTools.Fitting.LinearFitSE(x,dx,ref a,ref b, ref range, ref erry, ref erra, ref errb, ref ccor);
			if(isTop) XTopShrink = 1.0 + a;
			else XBotShrink = 1.0 + a;

			NumericalTools.Fitting.LinearFitSE(y,dy,ref a,ref b, ref range, ref erry, ref erra, ref errb, ref ccor);
			if(isTop) YTopShrink = 1.0 + a;
			else YBotShrink = 1.0 + a;

			return ComputationResult.OK;
		}

		public void Check(string inputrwc, LinkedZone lz, string output)
		{
			Catalog Cat = null;					
			System.IO.FileStream f = null;
			try
			{
				f = new System.IO.FileStream(inputrwc, System.IO.FileMode.Open, System.IO.FileAccess.Read);
				Cat = new Catalog(f);
			}
			catch (Exception x)
			{
				throw x;
			}
			finally 
			{
				f.Close();
			}

			//TODO: serve??? if(File.Exists(output)) Read(output);
			Attempts++;

			int XSize = (int)Cat.XSize;
			int YSize = (int)Cat.YSize;

            //XFrag = Cat[0,XSize-1];
            //YFrag = (uint)(Cat.Fragments/XFrag + 0.5);
			
			double fArea = 
				(Cat.Extents.MaxX-Cat.Extents.MinX)*(Cat.Extents.MaxY-Cat.Extents.MinY)/Cat.Fragments;

			double [] FragTracks = new double[Cat.Fragments];

			int i;
			for (i=0; i<lz.Length; i++)
			{
				MIPIndexedEmulsionTrack top = new MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Top,i);
				MIPIndexedEmulsionTrack bot = new MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Bottom,i);
				FragTracks[top.OriginalRawData.Fragment-1]++;
			}

			ComputeShrinkage(lz, true);
			ComputeShrinkage(lz, false);

			MakeFragmentCheckArray(Cat.Fragments);
			FragmentCheck fc;

			for (i = 0; i < Cat.Fragments; i++)	
			{
				fc = (FragCheck[i] = new FragmentCheck());
				fc.Index = i;
				fc.Density = FragTracks[i]/fArea;
			}

			Write(output);
		}

		public static bool IsScannedStripGood(string checkfile, double minDensityBase)
		{
			StatusZone sinfo = new StatusZone();
			sinfo.Read(checkfile);
			FragmentCheck[] fcs = (FragmentCheck[])sinfo.FragCheck;
			foreach (FragmentCheck fc in fcs)
			{
				if (fc.Density<minDensityBase)
				{
					return false;
				}
			}
			return true;
		}
	}

    [Serializable]
    public class StripMonitorArrayClass
    {
        private int Length;
        private StripMonitor[] Array;

        public StripMonitorArrayClass(int n)
        {
            Length = n;
            Array = new StripMonitor[n];
        }

        public StripMonitor this[int index]
        {
            get
            {
                return Array[index];
            }
            set
            {
                Array[index] = value;
            }
        }

        public int Count
        {
            get
            {
                int n = 0;
                for (int i = 0; i < Length; i++) if (Array[i] != null) n++;
                return n;
            }
        }
    }

    [Serializable]
    public class StripMonitor
    {
        public int Id;
        public int Nbasetracks;

        public DateTime StartTime;
        public DateTime EndTime;

        public uint Length;
        public ViewMonitor[] Views;

        internal static XmlSerializer ser = new XmlSerializer(typeof(StripMonitor));

        public StripMonitor()
        {
            Length = 0;
            Views = null;
        }
        public StripMonitor(int nviews)
        {
            Length = 0;
            Views = new ViewMonitor[nviews];
        }
        private void Write(string filename)
        {
            System.IO.FileStream fs = new System.IO.FileStream(filename, System.IO.FileMode.Create);
            try
            {
                ser.Serialize(fs, this);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
        }

        public void Read(string filename)
        {
            StripMonitor info = null;
            System.IO.StreamReader fs = new StreamReader(filename);
            //TODO: need executable permission System.IO.FileStream fs = new System.IO.FileStream(filename, System.IO.FileMode.Open);
            try
            {
                info = (StripMonitor)ser.Deserialize(fs);
            }
            catch (SerializationException e)
            {
                if (fs != null) fs.Close();
                Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                Copy(info);
                fs.Close();
            }
            return;
        }

        public void Copy(StripMonitor info)
        {
            Id = info.Id;
            Length = info.Length;
            StartTime = info.StartTime;
            EndTime = info.EndTime;
            Nbasetracks = info.Nbasetracks;

            if (info.Views != null)
            {
                Views = new ViewMonitor[info.Views.Length];
                int i;
                for (i = 0; i < Views.Length; i++) Views[i] = info.Views[i];
            }
            else Views = null;
            return;
        }


        public void Fill(int series, string inputrwc, string inputtlg, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, string output)
        {
            Id = series;

            StartTime = System.IO.File.GetCreationTime(inputrwc);
            EndTime = System.IO.File.GetCreationTime(inputtlg);

            if (lz == null) lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(inputrwc.Replace(".rwc", "_sel.tlg"), typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
            //Nbasetracks = lz.Length;
            Nbasetracks = 0;

            for (int i = 0; i < lz.Length; i++)
            {
                SySal.Scanning.MIPBaseTrack t = lz[i];
                if (t.Info.Sigma >= 0)
                    Nbasetracks++;
            }

            string BaseName = inputrwc.Substring(0, inputrwc.Length - 4);

            System.IO.FileStream f = null;
            try
            {
                f = new System.IO.FileStream(BaseName + ".rwc", System.IO.FileMode.Open, System.IO.FileAccess.Read); //Raw
            }
            catch (Exception x)
            {
                throw x;
            }
            SySal.Scanning.Plate.IO.OPERA.RawData.Catalog Cat = new SySal.Scanning.Plate.IO.OPERA.RawData.Catalog(f);
            f.Close();

            int YSize = (int)Cat.YSize;
            int XSize = (int)Cat.XSize;

            uint start = Cat[0, 0];
            uint end = start;
            for (int ix = 0; ix < Cat.XSize; ix++)
                for (int iy = 0; iy < Cat.YSize; iy++)
                    end = Math.Max(end, Cat[iy, ix]);

            double stepx = Cat.Steps.X;
            double stepy = Cat.Steps.Y;

            uint VInactiveLayers=0, VLayers=0;
            double NptMinV=0, NptMinH=0, NptMin01=0;

            for (int i = 0; i < Cat.SetupInfo.Length; i++)
            {
                for (int j = 0; j < Cat.SetupInfo[i].Length; j++)
                {
                    if (String.Compare(Cat.SetupInfo[i][j].Name, "VInactiveLayers") == 0)
                        VInactiveLayers = Convert.ToUInt32(Cat.SetupInfo[i][j].Value.ToString(System.Globalization.CultureInfo.InvariantCulture));

                    else if (String.Compare(Cat.SetupInfo[i][j].Name, "VLayers") == 0)
                        VLayers = Convert.ToUInt32(Cat.SetupInfo[i][j].Value.ToString(System.Globalization.CultureInfo.InvariantCulture));

                    else if (String.Compare(Cat.SetupInfo[i][j].Name, "NptMinV") == 0)
                        NptMinV = Convert.ToDouble(Cat.SetupInfo[i][j].Value, System.Globalization.CultureInfo.InvariantCulture);

                    else if (String.Compare(Cat.SetupInfo[i][j].Name, "NptMin01") == 0)
                        NptMin01 = Convert.ToDouble(Cat.SetupInfo[i][j].Value, System.Globalization.CultureInfo.InvariantCulture);

                    else if (String.Compare(Cat.SetupInfo[i][j].Name, "NptMinH") == 0)
                        NptMinH = Convert.ToDouble(Cat.SetupInfo[i][j].Value, System.Globalization.CultureInfo.InvariantCulture);
                }
            }

            uint mingrains = Convert.ToUInt32(Math.Ceiling(Math.Min(NptMinV, Math.Min(NptMin01, NptMinH))));
            uint maxgrains = VLayers - VInactiveLayers;

            //TODO: should be possible to remove since the previuos lines are correct
            if (mingrains > maxgrains) mingrains = 6;

            
            Length = (uint)(XSize * YSize);
            Views = new ViewMonitor[Length];

            int topViews = 0;
            int emptyTopViews = 0;
            int botViews = 0;
            int emptyBotViews = 0;

            int counter = 0; //contatore delle view per strip

            System.IO.FileStream ff = null;
            SySal.Scanning.Plate.IO.OPERA.RawData.Fragment Frag = null;

            for (uint j = start; j <= end; j++)
            {
                string fname = BaseName + ".rwd." + System.Convert.ToString(j, 16).PadLeft(8, '0');
                try
                {
                    ff = new System.IO.FileStream(fname, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                    Frag = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment(ff);
                }
                catch { }
                finally
                {
                    if (ff != null)
                        ff.Close();
                }

                for (int k = 0; k < Frag.Length; k++)
                {
                    this[counter] = new ViewMonitor(mingrains, maxgrains);
                    //TODO                    this[counter].Id = (Id - 1) * end * Frag.Length + (Frag.Index - 1) * Frag.Length + k + 1;
                    this[counter].Id = (Frag.Index - 1) * Frag.Length + k + 1;
                    this[counter].X = Frag[k].Top.MapPos.X;
                    this[counter].Y = Frag[k].Top.MapPos.Y;
                    this[counter].TopTracks = Frag[k].Top.Length;
                    this[counter].BotTracks = Frag[k].Bottom.Length;
                    this[counter].Z = Frag[k].Top.BottomZ;
                    this[counter].PlasticBaseThickness = Frag[k].Top.BottomZ - Frag[k].Bottom.TopZ;
                    this[counter].TopLayerThickness = Frag[k].Top.TopZ - Frag[k].Top.BottomZ;
                    this[counter].BottomLayerThickness = Frag[k].Bottom.TopZ - Frag[k].Bottom.BottomZ;

                    for (int l = 0; l < Frag[k].Top.Length; l++)
                        this[counter].TopTrackGrainCounter[Frag[k].Top[l].Info.Count - mingrains]++;

                    for (int l = 0; l < Frag[k].Bottom.Length; l++)
                        this[counter].BottomTrackGrainCounter[Frag[k].Bottom[l].Info.Count - mingrains]++;

                    topViews++;
                    if (Frag[k].Top.Length == 0) emptyTopViews++;

                    botViews++;
                    if (Frag[k].Bottom.Length == 0) emptyBotViews++;

                    counter++;
                }
                Write(output);
            }
        }

        public ViewMonitor this[int index]
        {
            get
            {
                return Views[index];
            }
            set
            {
                Views[index] = value;
            }
        }
    }

    [Serializable]
    public class ViewMonitor
    {
        public long Id;
        public double X;
        public double Y;
        public double Z;
        public int TopTracks;
        public int BotTracks;

        public double TopLayerThickness;
        public double BottomLayerThickness;
        public double PlasticBaseThickness;

        public uint MinGrains;
        public uint MaxGrains;

        public int[] TopTrackGrainCounter = null;
        public int[] BottomTrackGrainCounter = null;

        public ViewMonitor() { }

        public ViewMonitor(uint mingrains, uint maxgrains)
        {
            MinGrains = mingrains;
            MaxGrains = maxgrains;

            TopTrackGrainCounter = new int[MaxGrains - MinGrains + 1];
            BottomTrackGrainCounter = new int[MaxGrains - MinGrains + 1];
        }
    }

    [Serializable]
    public class StripStatus
    {
        public int Id;
        public bool Scanned;
        public bool Monitored;
        public string MonitoringFile;
        public bool Processed;
        public bool Completed;
        public uint MaxTrials;

        public StripStatus()
        {
            ;
        }
        public StripStatus(int id, uint maxtrials)
        {
            Id = id;
            MaxTrials = maxtrials;
            Scanned = false;
            Monitored = false;
            Processed = false;
            Completed = false;
            MonitoringFile = "";
        }
    }

    [Serializable]
    public class StripStatusArrayClass
    {
        public int Length;
        public StripStatus[] Array;

        public StripStatusArrayClass(int n, uint maxtrials)
        {
            Length = n;
            Array = new StripStatus[n];
            for (int i = 0; i < Length; i++) Array[i] = new StripStatus(i + 1, maxtrials);
        }

        public StripStatus this[int index]
        {
            get
            {
                return Array[index];
            }
            set
            {
                Array[index] = value;
            }
        }
    }

    /// <summary>
    /// Quality check for a strip.
    /// </summary>
    [Serializable]
    public class StripLinkStatusInfo
    {
        public int Attempts;
        public double XTopShrink;
        public double YTopShrink;
        public double XBotShrink;
        public double YBotShrink;

        //internal static uint XFrag, YFrag;

        public FragmentCheck[] FragCheck;

        internal static XmlSerializer ser = new XmlSerializer(typeof(StripLinkStatusInfo));

        public StripLinkStatusInfo()
        {
            Attempts = 0;
            XTopShrink = .0;
            YTopShrink = .0;
            XBotShrink = .0;
            YBotShrink = .0;

            FragCheck = null;
        }

        public StripLinkStatusInfo(StripLinkStatusInfo info)
        {
            Attempts = info.Attempts;
            XTopShrink = info.XTopShrink;
            YTopShrink = info.YTopShrink;
            XBotShrink = info.XBotShrink;
            YBotShrink = info.YBotShrink;
            FragCheck = new FragmentCheck[info.FragCheck.Length];
            int i;
            for (i = 0; i < FragCheck.Length; i++) FragCheck[i] = info.FragCheck[i];
        }

        public void Copy(StripLinkStatusInfo info)
        {
            this.Attempts = info.Attempts;
            this.XTopShrink = info.XTopShrink;
            this.YTopShrink = info.YTopShrink;
            this.XBotShrink = info.XBotShrink;
            this.YBotShrink = info.YBotShrink;
            if (info.FragCheck != null)
            {
                this.FragCheck = new FragmentCheck[info.FragCheck.Length];
                int i;
                for (i = 0; i < FragCheck.Length; i++) this.FragCheck[i] = info.FragCheck[i];
            }
            else FragCheck = null;
            return;
        }

        public FragmentCheck this[int index]
        {
            get
            {
                return FragCheck[index];
            }
            set
            {
                FragCheck[index] = value;
            }
        }

        public void MakeFragmentCheckArray(uint n)
        {
            FragCheck = new FragmentCheck[n];
        }

        private void Write(string filename)
        {
            FileStream fs = new FileStream(filename, FileMode.Create);
            try
            {
                ser.Serialize(fs, this);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
        }

        public void Read(string filename)
        {
            StripLinkStatusInfo info = null;
            FileStream fs = new FileStream(filename, FileMode.Open);
            try
            {
                info = (StripLinkStatusInfo)ser.Deserialize(fs);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                this.Copy(info);
                fs.Close();
            }
            return;
        }

        private ComputationResult ComputeShrinkage(LinkedZone lz, bool isTop)
        {
            int NData = lz.Length;
            double[] x = new double[NData];
            double[] dx = new double[NData];
            double[] y = new double[NData];
            double[] dy = new double[NData];

            int i;
            for (i = 0; i < NData; i++)
            {
                if (isTop) x[i] = lz.Top[i].Info.Slope.X;
                else x[i] = lz.Bottom[i].Info.Slope.X;
                dx[i] = lz[i].Info.Slope.X - x[i];
                if (isTop) y[i] = lz.Top[i].Info.Slope.Y;
                else y[i] = lz.Bottom[i].Info.Slope.Y;
                dy[i] = lz[i].Info.Slope.Y - y[i];
            }

            double a = 0, b = 0, range = 0, erry = 0, erra = 0, errb = 0, ccor = 0;

            NumericalTools.Fitting.LinearFitSE(x, dx, ref a, ref b, ref range, ref erry, ref erra, ref errb, ref ccor);
            if (isTop) XTopShrink = 1.0 + a;
            else XBotShrink = 1.0 + a;

            NumericalTools.Fitting.LinearFitSE(y, dy, ref a, ref b, ref range, ref erry, ref erra, ref errb, ref ccor);
            if (isTop) YTopShrink = 1.0 + a;
            else YBotShrink = 1.0 + a;

            return ComputationResult.OK;
        }

        public void Check(string inputrwc, LinkedZone lz, string output)
        {
            Catalog Cat = null;
            System.IO.FileStream f = null;
            try
            {
                f = new System.IO.FileStream(inputrwc, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                Cat = new Catalog(f);
            }
            catch (Exception x)
            {
                throw x;
            }
            finally
            {
                f.Close();
            }

            //TODO: serve??? if(File.Exists(output)) Read(output);
            Attempts++;

            int XSize = (int)Cat.XSize;
            int YSize = (int)Cat.YSize;

            //XFrag = Cat[0,XSize-1];
            //YFrag = (uint)(Cat.Fragments/XFrag + 0.5);

            double fArea =
                (Cat.Extents.MaxX - Cat.Extents.MinX) * (Cat.Extents.MaxY - Cat.Extents.MinY) / Cat.Fragments;

            double[] FragTracks = new double[Cat.Fragments];

            int i;
            for (i = 0; i < lz.Length; i++)
            {
                MIPIndexedEmulsionTrack top = new MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Top, i);
                MIPIndexedEmulsionTrack bot = new MIPIndexedEmulsionTrack((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz[i].Bottom, i);
                FragTracks[top.OriginalRawData.Fragment - 1]++;
            }

            ComputeShrinkage(lz, true);
            ComputeShrinkage(lz, false);

            MakeFragmentCheckArray(Cat.Fragments);
            FragmentCheck fc;

            for (i = 0; i < Cat.Fragments; i++)
            {
                fc = (FragCheck[i] = new FragmentCheck());
                fc.Index = i;
                fc.Density = FragTracks[i] / fArea;
            }

            Write(output);
        }

        public static bool IsScannedStripGood(string checkfile, double minDensityBase)
        {
            StripLinkStatusInfo sinfo = new StripLinkStatusInfo();
            sinfo.Read(checkfile);
            FragmentCheck[] fcs = (FragmentCheck[])sinfo.FragCheck;
            foreach (FragmentCheck fc in fcs)
            {
                if (fc.Density < minDensityBase)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
