using System;
using SySal;
using System.Xml;
using System.Xml.Serialization;
using SySal.Processing;

namespace SySal.Executables.BatchFragmentShiftCorrect
{
	/// <summary>
	/// Batch fragment shift correction configuration.
	/// </summary>
	[Serializable]
	[XmlType("BatchFragmentShiftCorrect.Config")]
	public class Config
	{
		public SySal.Processing.FragShiftCorrection.Configuration FragmentShiftCorrectionConfig;
	}

	/// <summary>
	/// Reads RWD files from a specified catalog (RWC) and computes a set of correction parameters for systematic shift errors. The correction is just computed, not applied.
	/// </summary>
	/// <remarks>
	/// <para>BatchFragmentShiftCorrect uses SySal.Processing.FragShiftCorrection.</para>
	/// <para>
	/// The syntax for the command line is:
	/// <code>
	/// BatchFragmentShiftCorrect.exe &lt;input RWC path&gt; &lt;output XML file path&gt; &lt;XML config Opera persistence path&gt;
	/// </code>
	/// Notice the last parameter is an OPERA persistence path, i.e. a local file, network file or DB configuration (e.g. <c>db:\1293.xml</c>).
	/// </para>
	/// <para>
	/// Usage example (command line):
	/// <example>
	/// <c>BatchFragmentShiftCorrect.exe c:\myset.rwc c:\myoutputparams.xml c:\correctionconfig.xml</c>
	/// </example>
	/// </para>
	/// <para>
	/// The syntax of the configuration file for correction computation is:
	/// <code>
	/// &lt;BatchFragmentShiftCorrect.Config&gt;
	///  &lt;FragmentShiftCorrectionConfig&gt;
	///   &lt;Name&gt;Default Fragment Field Shift Manager Config&lt;/Name&gt;
	///   &lt;MinGrains&gt;6&lt;/MinGrains&gt;
	///   &lt;MinSlope&gt;0.01&lt;/MinSlope&gt;
	///   &lt;MergePosTol&gt;20&lt;/MergePosTol&gt;
	///   &lt;MergeSlopeTol&gt;0.02&lt;/MergeSlopeTol&gt;
	///   &lt;PosTol&gt;50&lt;/PosTol&gt;
	///   &lt;SlopeTol&gt;0.07&lt;/SlopeTol&gt;
	///   &lt;MinMatches&gt;2&lt;/MinMatches&gt;
	///   &lt;MaxMatchError&gt;1&lt;/MaxMatchError&gt;
	///   &lt;GrainsOverlapRatio&gt;0.2&lt;/GrainsOverlapRatio&gt;
	///   &lt;OverlapTol&gt;40&lt;/OverlapTol&gt;
	///   &lt;GrainZTol&gt;2&lt;/GrainZTol&gt;
	///   &lt;IsStep&gt;true&lt;/IsStep&gt;
	///   &lt;EnableHysteresis&gt;false&lt;/EnableHysteresis&gt;
	///  &lt;/FragmentShiftCorrectionConfig&gt;
	/// &lt;/BatchFragmentShiftCorrect.Config&gt;
	/// </code>
	/// See <see cref="SySal.Processing.FragShiftCorrection.FragmentShiftManager"/> and <see cref="SySal.Processing.FragShiftCorrection.LinearFragmentCorrectionWithHysteresis"/> for an explanation of the parameters.
	/// </para>
	/// </remarks>
	public class Exe
	{
		double LastPercent = 0.0;
		string BaseName = "";

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			if (args.Length != 3)
			{
				Console.WriteLine("BatchFragmentShiftCorrect - computes corrections for fragment shifts from RWD files.");
				Console.WriteLine("usage: BatchFragmentShiftCorrect <input RWC path> <output XML file path> <XML config Opera persistence path>");
				Console.WriteLine("XML config syntax:");
				BatchFragmentShiftCorrect.Config C = new BatchFragmentShiftCorrect.Config();
				C.FragmentShiftCorrectionConfig = (SySal.Processing.FragShiftCorrection.Configuration)new SySal.Processing.FragShiftCorrection.FragmentShiftManager().Config;
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchFragmentShiftCorrect.Config));
				System.IO.StringWriter ss = new System.IO.StringWriter();
				xmls.Serialize(ss, C);
				Console.WriteLine(ss.ToString());
				ss.Close();
				return;
			}
			string xmlconfig = ((SySal.OperaDb.ComputingInfrastructure.ProgramSettings)SySal.OperaPersistence.Restore(args[2], typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings))).Settings;
			Console.WriteLine(xmlconfig);
			Exe exec = new Exe();
//			try
			{
				exec.ProcessData(args[0], args[1], xmlconfig);
				Console.WriteLine("OK!");
			}
/*			catch (Exception x)
			{
				Console.Error.WriteLine(x.Message);
			}
*/		}

		void Progress(double percent)
		{
			if (percent - LastPercent >= 10.0)
			{
				Console.WriteLine("{0}%", (int)percent);
				LastPercent = percent;
			}
		}

		protected SySal.Scanning.Plate.IO.OPERA.RawData.Fragment LoadFragment(uint index)
		{
			System.IO.FileStream f = new System.IO.FileStream(BaseName + ".rwd." + Convert.ToString(index, 16).PadLeft(8, '0'), System.IO.FileMode.Open, System.IO.FileAccess.Read);
			SySal.Scanning.Plate.IO.OPERA.RawData.Fragment Frag = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment(f);
			f.Close();
			return Frag;
		}

		void ProcessData(string input, string output, string programsettings)
		{
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Config));
			Config C = (Config)xmls.Deserialize(new System.IO.StringReader(programsettings));			
			SySal.Processing.FragShiftCorrection.FragmentShiftManager FSM = new SySal.Processing.FragShiftCorrection.FragmentShiftManager();
			FSM.Config = C.FragmentShiftCorrectionConfig;
			xmls.Serialize(Console.Out, C);
			Console.WriteLine();

			BaseName = input.ToLower().EndsWith(".rwc") ? (input.Substring(0, input.Length - 4)) : input;

			System.IO.FileStream f = new System.IO.FileStream(input, System.IO.FileMode.Open, System.IO.FileAccess.Read);
			SySal.Scanning.Plate.IO.OPERA.RawData.Catalog Cat = new SySal.Scanning.Plate.IO.OPERA.RawData.Catalog(f);
			f.Close();
			
			LastPercent = 0.0;
			FSM.Progress = new SySal.Scanning.PostProcessing.FieldShiftCorrection.dProgress(Progress);
			FSM.Load = new SySal.Scanning.PostProcessing.FieldShiftCorrection.dLoad(LoadFragment);

			SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShift [] shifts;
			SySal.Scanning.PostProcessing.FieldShiftCorrection.FragmentCorrection Corr;
			FSM.ComputeFragmentCorrection(Cat, SySal.Scanning.PostProcessing.FieldShiftCorrection.FieldShift.SideValue.Both, out shifts, out Corr);
			SySal.Processing.FragShiftCorrection.LinearFragmentCorrectionWithHysteresis LHCorr = (SySal.Processing.FragShiftCorrection.LinearFragmentCorrectionWithHysteresis)Corr;
			
			System.IO.StreamWriter w = new System.IO.StreamWriter(output, false);
			xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Processing.FragShiftCorrection.LinearFragmentCorrectionWithHysteresis));
			xmls.Serialize(w, LHCorr);
			w.Flush();
			w.Close();

			Progress(100.0);

			GC.Collect();
			Console.WriteLine("Result written to: " + output);
		}

	}
}
