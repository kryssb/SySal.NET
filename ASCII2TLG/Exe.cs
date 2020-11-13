using System;

namespace SySal.Executables.ASCII2TLG
{
	class LinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
	{		
		internal class View : SySal.Scanning.Plate.IO.OPERA.LinkedZone.View
		{
			internal View()
			{				
			}

			internal void Set(Side sd, SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack [] tks, double topz, double bottomz)
			{
				this.m_Position.X = 0.0;
				this.m_Position.Y = 0.0;
				this.m_Side = sd;
				this.m_BottomZ = bottomz;
				this.m_TopZ = topz;
				this.m_Tracks = tks;
			}
		}

		internal class Side : SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side
		{
			internal Side(SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack [] tks, double topz, double bottomz)
			{
				this.m_Views = new SySal.Scanning.Plate.IO.OPERA.LinkedZone.View[1];
				View vw = new View();
				vw.Set(this, tks, topz, bottomz);
				m_Views[0] = vw;
				this.m_Tracks = tks;
				this.m_TopZ = topz;
				this.m_BottomZ = bottomz;
			}
		}

		class MIPIndexedEmulsionTrack : SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack
		{
			internal MIPIndexedEmulsionTrack(int id, SySal.Tracking.MIPEmulsionTrackInfo info, bool istop, View vw)
			{				
				m_Grains = null;
				m_Id = id;
				m_Info = (SySal.Tracking.MIPEmulsionTrackInfo)info.Clone();
				m_Info.Intercept.X += m_Info.Slope.X * ((istop ? -200.0 : 0.0) - m_Info.Intercept.Z);
				m_Info.Intercept.Y += m_Info.Slope.Y * ((istop ? -200.0 : 0.0) - m_Info.Intercept.Z);
				m_Info.Intercept.Z = istop ? -200.0 : 0.0;
				m_Info.TopZ = istop ? 43.0 : -200.0;
				m_Info.BottomZ = istop ? 0.0 : -243.0;
				m_OriginalRawData.Fragment = 0;
				m_OriginalRawData.View = 0;
				m_OriginalRawData.Track = id;				
				m_View = vw;
			}
		}

		class MIPBaseTrack : SySal.Scanning.MIPBaseTrack
		{
			internal MIPBaseTrack(int id, SySal.Tracking.MIPEmulsionTrackInfo info, MIPIndexedEmulsionTrack top, MIPIndexedEmulsionTrack bottom)
			{
				m_Id = id;
				m_Info = info;
				m_Top = top;
				m_Bottom = bottom;
			}
		}

		public static readonly string DisplayFormat = "ASCII format: <pts> <areasum> <px> <py> <sx> <sy> <sigma>";

		public LinkedZone(System.IO.StreamReader r)
		{
			this.m_Transform.MXX = this.m_Transform.MYY = 1.0; 
			this.m_Transform.MXY = this.m_Transform.MYX = 0.0;
			this.m_Transform.TX = this.m_Transform.TY = this.m_Transform.TZ = 0.0;
			this.m_Transform.RX = this.m_Transform.RY = 0.0;
			System.Collections.ArrayList ar = new System.Collections.ArrayList();
			string line;
			while (((line = r.ReadLine()) != null) && ((line = line.Trim()) != ""))
			{
				int pos = 0;
				SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
				info.Count = Convert.ToUInt16(GetNextToken(line, ref pos));
				info.Field = 0;
				info.AreaSum = Convert.ToUInt32(GetNextToken(line, ref pos));
				info.Intercept.X = Convert.ToDouble(GetNextToken(line, ref pos));
				info.Intercept.Y = Convert.ToDouble(GetNextToken(line, ref pos));
				info.Intercept.Z = 0.0;
				info.Slope.X = Convert.ToDouble(GetNextToken(line, ref pos));
				info.Slope.Y = Convert.ToDouble(GetNextToken(line, ref pos));
				info.Slope.Z = 1.0;
				info.Sigma = Convert.ToDouble(GetNextToken(line, ref pos));
				info.TopZ = 43.0;
				info.BottomZ = -243.0;				
				ar.Add(info);
			}
			int i;
			MIPIndexedEmulsionTrack [] ttkarr = new MIPIndexedEmulsionTrack[ar.Count];
			m_Top = new Side(ttkarr, 43.0, 0.0);
			for (i = 0; i < ttkarr.Length; i++)
				ttkarr[i] = new MIPIndexedEmulsionTrack(i, (SySal.Tracking.MIPEmulsionTrackInfo)ar[i], true, (View)((Side)m_Top).View(0));
			MIPIndexedEmulsionTrack [] btkarr = new MIPIndexedEmulsionTrack[ar.Count];
			m_Bottom = new Side(btkarr, -200.0, -243.0);
			for (i = 0; i < btkarr.Length; i++)
				btkarr[i] = new MIPIndexedEmulsionTrack(i, (SySal.Tracking.MIPEmulsionTrackInfo)ar[i], false, (View)((Side)m_Bottom).View(0));
			m_Tracks = new MIPBaseTrack[ar.Count];
			for (i = 0; i < m_Tracks.Length; i++)
				m_Tracks[i] = new MIPBaseTrack(i, (SySal.Tracking.MIPEmulsionTrackInfo)ar[i], ttkarr[i], btkarr[i]);
		}

		static string GetNextToken(string s, ref int pos)
		{
			string n = "";
			while (pos < s.Length && (s[pos] == ' ' || s[pos] == '\t')) n += s[pos++];
			while (pos < s.Length && (s[pos] != ' ' && s[pos] != '\t')) n += s[pos++];
			return n;
		}
	}

	/// <summary>
	/// Converts an ASCII file of n-tuples to a TLG file.
	/// </summary>
	/// <remarks>
	/// <para>The n-tuples must be in the format: <c>pts areasum px py sx sy sigma</c></para>
	/// <para>
	/// Usage example (command line):
	/// <example>
	/// <c>ASCII2TLG.exe c:\myfile.txt c:\myfile.tlg</c>
	/// </example>
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
			//
			// TODO: Add code to start application here
			//
			if (args.Length != 2)
			{
				Console.WriteLine("usage: ASCII2TLG <input ASCII file> <output TLG file>");
				Console.WriteLine(LinkedZone.DisplayFormat);
				return;
			}
			System.IO.StreamReader r = new System.IO.StreamReader(args[0]);
			LinkedZone lz = new LinkedZone(r);
			r.Close();
			System.IO.FileStream w = new System.IO.FileStream(args[1], System.IO.FileMode.Create, System.IO.FileAccess.Write);
			lz.Save(w);
			w.Flush();
			w.Close();
		}
	}
}
