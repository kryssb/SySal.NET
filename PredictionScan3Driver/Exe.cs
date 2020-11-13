using System;
using SySal;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal.DAQSystem.Drivers;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{

    /// <summary>
    /// Intercalibration modes.
    /// </summary>
    [Serializable]
    public enum IntercalibrationMode
    {
        /// <summary>
        /// No local intercalibration is performed.
        /// </summary>
        None,
        /// <summary>
        /// Local rototranslation and Z displacement are computed.
        /// </summary>
        RotoTranslationDeltaZ,
        /// <summary>
        /// Local affine transformation and Z displacement are computed.
        /// <b>NOTICE: This mode is not yet implemented.</b>
        /// </summary>
        AffineDeltaZ
    }

    /// <summary>
    /// Settings for PredictionScan3Driver.
    /// </summary>
    [Serializable]
    public class PredictionScan3Settings
    {
        /// <summary>
        /// The Id of the scanning program settings.
        /// </summary>
        public long ScanningConfigId;
        /// <summary>
        /// The Id of the linking program settings.
        /// </summary>
        public long LinkConfigId;
        /// <summary>
        /// The Id for quality cut.
        /// </summary>
        public long QualityCutId;
        /// <summary>
        /// Slope tolerance for candidate selection (transverse component and tolerance at zero slope).
        /// </summary>
        public double SlopeTolerance;
        /// <summary>
        /// Slope tolerance increase factor. The actual longitudinal tolerance is "SlopeTolerance + Slope * SlopeToleranceIncreaseWithSlope".
        /// </summary>
        public double SlopeToleranceIncreaseWithSlope;
        /// <summary>
        /// Position tolerance for candidate selection (transverse component and tolerance at zero slope). 
        /// </summary>
        public double PositionTolerance;
        /// <summary>
        /// Position tolerance increase factor. The actual longitudinal tolerance is "PositionTolerance + Slope * PositionToleranceIncreaseWithSlope".
        /// </summary>
        public double PositionToleranceIncreaseWithSlope;
        /// <summary>
        /// Expected base thickness. This is used to set the center of the field of view halfway between the two microtracks of the scanback track.
        /// </summary>
        public double BaseThickness;
        /// <summary>
        /// If true, the tolerance settings in this configuration are ignored, and tolerances are read from the prediction table.
        /// </summary>
        public bool ReadTolerancesFromPredictionTable;
        /// <summary>
        /// The selection function to be minimized to select the candidate.
        /// </summary>
        public string SelectionFunction;
        /// <summary>
        /// Minimum acceptable bound for the value of the selection function.
        /// </summary>
        public double SelectionFunctionMin;
        /// <summary>
        /// Maximum acceptable bound for the value of the selection function.
        /// </summary>
        public double SelectionFunctionMax;
        /// <summary>
        /// Maximum number of trials to search the candidate.
        /// </summary>
        public uint MaxTrials;
        /// <summary>
        /// SQL query to select recalibration tracks. If left empty or null, no recalibration is done. The query must return the following fields (in the exact order): <c>POSX POSY SLOPEX SLOPEY</c>. 
        /// The query can use <c>_BRICK_</c> and <c>_PLATE_</c> to denote the current brick and the current plate.
        /// </summary>
        public string RecalibrationSelectionText;
        /// <summary>
        /// If set to a value different than <c>None</c>, local intercalibration around average prediction position is performed. In this case, <c>RecalibrationSelectionText</c> is overridden. 
        /// The size of the area to be used for intercalibration is defined by <c>RecalibrationMinXDistance</c> and <c>RecalibrationMinYDistance</c>. 
        /// The position tolerance used for pattern matching is <c>RecalibrationPosTolerance</c>. Pattern matching is considered successful if at least <c>RecalibrationMinTracks</c> matches are found.
        /// </summary>
        public IntercalibrationMode LocalIntercalibration;
        /// <summary>
        /// This array is used when <c>LocalIntercalibration</c> is different from <c>None</c>. It contains the list of test values for DeltaZ difference from the nominal value.
        /// <example>If 		
        /// <code>LocalDeltaZ =
        /// &lt;LocalDeltaZ&gt;
        ///  &lt;double&gt;100&lt;/double&gt;
        ///  &lt;double&gt;0&lt;/double&gt;
        ///  &lt;double&gt;-100&lt;/double&gt;
        /// &lt;/LocalDeltaZ&gt;</code>
        /// and the nominal DeltaZ is 1300, intercalibration will be attempted with DeltaZ = 1200, 1300, 1400.</example>
        /// </summary>
        public double[] LocalDeltaZ;
        /// <summary>
        /// This array is used when <c>LocalIntercalibration</c> is different from <c>None</c>. It contains the list of values for DeltaZ for alignments to be excluded.
        /// <example>If 
        /// <code>LocalExcludedDeltaZ = 
        /// &lt;LocalExcludedDeltaZ&gt;
        ///  &lt;double&gt;300&lt;/double&gt;
        /// &lt;/LocalExcludedDeltaZ&gt;</code>
        /// then tracks that map with DeltaZ = 300 will be excluded from intercalibration (this is the case for transport alignment in OPERA).</example>
        /// </summary>
        public double[] LocalExcludedDeltaZ;
        /// <summary>
        /// The Id of the linking program settings for local intercalibration (ignored if <c>LocalIntercalibration</c> = <c>None</c>).
        /// </summary>
        public long LocalIntercalibrationLinkConfigId;
        /// <summary>
        /// The Id for quality cut for local intercalibration (ignored if <c>LocalIntercalibration</c> = <c>None</c>).
        /// </summary>
        public long LocalIntercalibrationQualityCutId;
        /// <summary>
        /// Maximum slope accepted for intercalibration. (ignored if <c>LocalIntercalibration</c> = <c>None</c>).
        /// </summary>
        public double LocalIntercalibrationMaxSlope;
        /// <summary>
        /// Slope tolerance for intercalibration. (ignored if <c>LocalIntercalibration</c> = <c>None</c>).
        /// </summary>
        public double LocalIntercalibrationSlopeTolerance;
        /// <summary>
        /// The tracks selected for recalibration must span in the X direction at least a distance equal to RecalibrationMinXDistance. Ignored if no recalibration is to be done.
        /// </summary>
        public double RecalibrationMinXDistance;
        /// <summary>
        /// The tracks selected for recalibration must span in the Y direction at least a distance equal to RecalibrationMinYDistance. Ignored if no recalibration is to be done.
        /// </summary>
        public double RecalibrationMinYDistance;
        /// <summary>
        /// Recalibration is successful if at least RecalibrationMinTracks are found. Ignored if no recalibration is to be done.
        /// </summary>
        public int RecalibrationMinTracks;
        /// <summary>
        /// Position tolerance to find recalibration candidates. Ignored if no recalibration is to be done.
        /// </summary>
        public double RecalibrationPosTolerance;
        /// <summary>
        /// When set to <c>true</c>, enables forking, i.e. yielding multiple candidates for a single prediction.
        /// </summary>
        public bool EnableForking;
        /// <summary>
        /// When set to <c>true</c>, tracks are searched using slope preselection.
        /// </summary>
        public bool EnableSlopePresetting;
        /// <summary>
        /// Tolerance on microtrack to be applied separately on each projection (X and Y), at slopeX/Y = 0, when slope presetting is enabled. This member is ignored if <c>EnableSlopePresetting</c> is <c>false</c>. 
        /// The resulting tolerance is <c>SlopePresetXYTolerance + SlopePresetXYToleranceIncreaseWithSlope * abs(Slope[X|Y])</c>.
        /// </summary>
        public double SlopePresetXYTolerance;
        /// <summary>
        /// Tolerance increase on microtrack to be applied separately on each projection (X and Y) when slope presetting is enabled. This member is ignored if <c>EnableSlopePresetting</c> is <c>false</c>. 
        /// The resulting tolerance is <c>SlopePresetXYTolerance + SlopePresetXYToleranceIncreaseWithSlope * abs(Slope[X|Y])</c>.
        /// </summary>
        public double SlopePresetXYToleranceIncreaseWithSlope;
        /// <summary>
        /// When this flag is set to <c>true</c>, if a scanback track is not found in automatic mode, at the end of the last trial a manual scanning is required. 
        /// <u>This requires operator presence during scanning.</u>
        /// </summary>
        public bool AskManualScanIfMissing;
        /// <summary>
        /// If <c>null</c> (the default), PredictionScan3Driver will write the results of the search to TB_SCANBACK_PREDICTIONS (if allowed by <see cref="AvoidWritingScanbackPredictions"/>). If <c>true</c>, the candidates will be written to a file in the <c>Scratch</c> 
        /// directory, named "AAAAAAA_XXXXXX_YY_ZZZZZZ.txt", where AAAAA stands for the value of ResultFile, XXXX for the brick number, YY for the plate number, and ZZZZZZZZ for the process operation id.
        /// </summary>
        /// <remarks>The syntax used for the result file is:
        /// <para><c>ID_EVENTBRICK ID_PLATE PATH ID_PROCESSOPERATION PPX PPY PSX PSY GRAINS AREASUM FPX FPY FSX FSY SIGMA DPX DPY DSX DSY</c></para></remarks>
        public string ResultFile;
        /// <summary>
        /// If <c>false</c>(default), the microscope is directed to go to the next prediction immediately after finishing the scanning. Set to <c>true</c> if this causes mechanical stresses or erratic motion.
        /// </summary>
        public bool DisableScanAndMoveToNext;
        /// <summary>
        /// If <c>true</c>, scanning data are written to the DB, but the TB_SCANBACK_PREDICTIONS table is not updated; if <c>false</c>, the TB_SCANBACK_PREDICTIONS table is updated with the found candidates.
        /// </summary>
        public bool AvoidWritingScanbackPredictions;
    }

    internal class IntercalTrack
    {
        public long IdZone;

        public int Id;

        public SySal.Tracking.MIPEmulsionTrackInfo Info;
    }

    internal class ForkInfo
    {
        public double SelFuncValue;

        public long CandidateId;

        public long ForkPathId;
    }

    internal class ForkInfoComparer : System.Collections.IComparer
    {
        public int Compare(object x, object y)
        {
            double c = ((ForkInfo)x).SelFuncValue - ((ForkInfo)y).SelFuncValue;
            if (c < 0) return -1;
            if (c > 0) return 1;
            return 0;
        }

        public static ForkInfoComparer TheForkInfoComparer = new ForkInfoComparer();
    }

    internal class ManualCheck
    {
        public SySal.DAQSystem.Scanning.ManualCheck.InputBaseTrack Input;

        public SySal.DAQSystem.Scanning.ManualCheck.OutputBaseTrack Output;

        public long ZoneId;

        public int NextTopMicroId;

        public int NextBottomMicroId;

        public int NextBaseId;

        public double BaseThickness;

        public long PredictionId;
    }

    internal class ScanningResult
    {
        public long PathId;

        public long CandidateId;

        public double PPX;

        public double PPY;

        public double PSX;

        public double PSY;

        public double DPX;

        public double DPY;

        public double DSX;

        public double DSY;

        public System.Collections.ArrayList ForkPaths;
    }

    internal class MyTransformation
    {
        public static void Transform(double X, double Y, ref double tX, ref double tY)
        {
            tX = Transformation.MXX * (X - Transformation.RX) + Transformation.MXY * (Y - Transformation.RY) + Transformation.TX + Transformation.RX;
            tY = Transformation.MYX * (X - Transformation.RX) + Transformation.MYY * (Y - Transformation.RY) + Transformation.TY + Transformation.RY;
        }

        public static void Deform(double X, double Y, ref double dX, ref double dY)
        {
            dX = Transformation.MXX * X + Transformation.MXY * Y;
            dY = Transformation.MYX * X + Transformation.MYY * Y;
        }

        public static SySal.DAQSystem.Scanning.IntercalibrationInfo Transformation = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
    }

    internal class LinkedZone : SySal.Scanning.Plate.IO.OPERA.LinkedZone
    {
        public class tMIPEmulsionTrack : SySal.Tracking.MIPEmulsionTrack
        {
            public static void ApplyTransformation(SySal.Tracking.MIPEmulsionTrack tk)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = tMIPEmulsionTrack.AccessInfo(tk);
                MyTransformation.Transform(info.Intercept.X, info.Intercept.Y, ref info.Intercept.X, ref info.Intercept.Y);
                MyTransformation.Deform(info.Slope.X, info.Slope.Y, ref info.Slope.X, ref info.Slope.Y);
                SySal.Tracking.Grain[] grains = tMIPEmulsionTrack.AccessGrains(tk);
                if (grains != null)
                    foreach (SySal.Tracking.Grain g in grains)
                        MyTransformation.Transform(g.Position.X, g.Position.Y, ref g.Position.X, ref g.Position.Y);
            }
        }

        public class tMIPBaseTrack : SySal.Scanning.MIPBaseTrack
        {
            public static void ApplyTransformation(SySal.Scanning.MIPBaseTrack tk)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = tMIPBaseTrack.AccessInfo(tk);
                MyTransformation.Transform(info.Intercept.X, info.Intercept.Y, ref info.Intercept.X, ref info.Intercept.Y);
                MyTransformation.Deform(info.Slope.X, info.Slope.Y, ref info.Slope.X, ref info.Slope.Y);

            }
        }

        public class Side : SySal.Scanning.Plate.Side
        {
            public static SySal.Tracking.MIPEmulsionTrack[] GetTracks(SySal.Scanning.Plate.Side s) { return Side.AccessTracks(s); }
        }

        public LinkedZone(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz)
        {
            MyTransformation.Transform(lz.Extents.MinX, lz.Extents.MinY, ref m_Extents.MinX, ref m_Extents.MinY);
            MyTransformation.Transform(lz.Extents.MaxX, lz.Extents.MaxY, ref m_Extents.MaxX, ref m_Extents.MaxY);
            MyTransformation.Transform(lz.Center.X, lz.Center.Y, ref m_Center.X, ref m_Center.Y);
            m_Id = lz.Id;
            m_Tracks = LinkedZone.AccessTracks(lz);
            m_Top = lz.Top;
            m_Bottom = lz.Bottom;
            foreach (SySal.Scanning.MIPBaseTrack btk in m_Tracks)
                tMIPBaseTrack.ApplyTransformation(btk);
            SySal.Tracking.MIPEmulsionTrack[] mutks;
            mutks = LinkedZone.Side.GetTracks(m_Top);
            foreach (SySal.Scanning.MIPIndexedEmulsionTrack mutk in mutks)
                tMIPEmulsionTrack.ApplyTransformation(mutk);
            mutks = LinkedZone.Side.GetTracks(m_Bottom);
            foreach (SySal.Scanning.MIPIndexedEmulsionTrack mutk in mutks)
                tMIPEmulsionTrack.ApplyTransformation(mutk);
            SySal.DAQSystem.Scanning.IntercalibrationInfo otr = lz.Transform;
            SySal.DAQSystem.Scanning.IntercalibrationInfo tr = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
            tr.MXX = MyTransformation.Transformation.MXX * otr.MXX + MyTransformation.Transformation.MXY * otr.MYX;
            tr.MXY = MyTransformation.Transformation.MXX * otr.MXY + MyTransformation.Transformation.MXY * otr.MYY;
            tr.MYX = MyTransformation.Transformation.MYX * otr.MXX + MyTransformation.Transformation.MYY * otr.MYX;
            tr.MYY = MyTransformation.Transformation.MYX * otr.MXY + MyTransformation.Transformation.MYY * otr.MYY;
            tr.RX = MyTransformation.Transformation.RX;
            tr.RY = MyTransformation.Transformation.RY;
            tr.TZ = MyTransformation.Transformation.TZ + otr.TZ;
            tr.TX = (tr.MXX - MyTransformation.Transformation.MXX) * (MyTransformation.Transformation.RX - otr.RX) + (tr.MXY - MyTransformation.Transformation.MXY) * (MyTransformation.Transformation.RY - otr.RY) + MyTransformation.Transformation.MXX * otr.TX + MyTransformation.Transformation.MXY * otr.TY + MyTransformation.Transformation.TX;
            tr.TY = (tr.MYX - MyTransformation.Transformation.MYX) * (MyTransformation.Transformation.RX - otr.RX) + (tr.MYY - MyTransformation.Transformation.MYY) * (MyTransformation.Transformation.RY - otr.RY) + MyTransformation.Transformation.MYX * otr.TX + MyTransformation.Transformation.MYY * otr.TY + MyTransformation.Transformation.TY;
            m_Transform = tr;
        }
    }

    /// <summary>
    /// PredictionScan3Driver executor.
    /// </summary>
    /// <remarks>
    /// <para>PredictionScan3Driver searches for predicted tracks.</para>
    /// <para>All input and output live in the DB. Predictions are read from TB_SCANBACK_PREDICTIONS and results are written to the same table.</para>
    /// <para>If needed, forking can be activated to provide multiple results for a single prediction search.</para>
    /// <para>
    /// Before real scanning, a recalibration can be computed using "safe" tracks.
    /// The selection must return the following columns, in this order:
    /// <c>POSX POSY SLOPEX SLOPEY</c>
    /// </para>
    /// <para>
    /// <c>_BRICK_</c> will be replaced with the brick ID.
    /// <c>_PLATE_</c> will be replaced with the plate ID.
    /// </para>
    /// <para>
    /// Supported Interrupts:
    /// <list type="bullet">
    /// <item>
    /// <description><c>IgnoreScanFailure False|True</c> instructs PredictionScan3Driver to stop on failed predictions or to skip them and go on.</description>
    /// </item>
    /// <item>
    /// <description><c>IgnoreRecalFailure False|True</c> instructs PredictionScan3Driver to stop on failed recalibration tracks or to skip them and go on.</description>
    /// </item>
    /// </list>
    /// Type: <c>PredictionScan3Driver /EasyInterrupt</c> for a graphical user interface to send interrupts.
    /// Type: <c>PredictionScan3Driver /CSBrick &lt;brick&gt; [&lt;candidate file&gt;]</c> for a graphical user interface to choose candidate tracks in the brick to connect to CS or TT predictions, or try to guess vertices.
    /// </para>
    /// <para>
    /// The following substitutions apply:
    /// <list type="table">
    /// <item><term><c>%EXEREP%</c></term><description>Executable repository path specified in the Startup file.</description></item>
    /// <item><term><c>%RWDDIR%</c></term><description>Output directory for Raw Data.</description></item>
    /// <item><term><c>%TLGDIR%</c></term><description>Output directory for linked zones.</description></item>
    /// <item><term><c>%RWD%</c></term><description>Scanning output file name (not including extension).</description></item>
    /// <item><term><c>%TLG%</c></term><description>Linked zone file name (not including extension).</description></item>
    /// <item><term><c>%SCRATCH%</c></term><description>Scratch directory specified in the Startup file.</description></item>
    /// <item><term><c>%ZONEID%</c></term><description>Hexadecimal file name for a zone.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// Known parameters for selection function:
    /// <list type="table">
    /// <item><term>PLATE</term><description>Current Plate.</description></item>
    /// <item><term>N</term><description>Grains.</description></item>
    /// <item><term>A</term><description>AreaSum.</description></item>
    /// <item><term>S</term><description>Sigma.</description></item>
    /// <item><term>PPX/Y</term><description>Predicted X/Y position.</description></item>
    /// <item><term>PSX/Y</term><description>Predicted X/Y slope.</description></item>
    /// <item><term>PSL </term><description>Predicted slope.</description></item>
    /// <item><term>FPX/Y</term><description>Found X/Y position.</description></item>
    /// <item><term>FSX/Y</term><description>Found X/Y slope.</description></item>
    /// <item><term>FSL</term><description>Found slope.</description></item>
    /// <item><term>TX/Y</term><description>X/Y component of transverse unit vector.</description></item>
    /// <item><term>LX/Y</term><description>X/Y component of longitudinal unit vector.</description></item>
    /// <item><term>DPX/Y</term><description>Found - Predicted X/Y position.</description></item>
    /// <item><term>DSX/Y</term><description>Found - Predicted X/Y slope.</description></item>
    /// <item><term>DPT/L</term><description>Found - Predicted Transverse/Longitudinal position.</description></item>
    /// <item><term>DST/L</term><description>Found - Predicted Transverse/Longitudinal slope.</description></item>
    /// <item><term>TRIALS</term><description>Number of remaining trials for the current prediction.</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// A sample XML configuration for PredictionScan3Driver follows:
    /// <example>
    /// <code>
    /// &lt;PredictionScan3Settings&gt;
    ///  &lt;ScanningConfigId&gt;1000000001587817&lt;/ScanningConfigId&gt;
    ///  &lt;LinkConfigId&gt;1000000000000018&lt;/LinkConfigId&gt;
    ///  &lt;QualityCutId&gt;1000000000000020&lt;/QualityCutId&gt;
    ///  &lt;SlopeTolerance&gt;0.03&lt;/SlopeTolerance&gt;
    ///  &lt;SlopeToleranceIncreaseWithSlope&gt;0.05&lt;/SlopeToleranceIncreaseWithSlope&gt;
    ///  &lt;PositionTolerance&gt;70&lt;/PositionTolerance&gt;
    ///  &lt;PositionToleranceIncreaseWithSlope&gt;6&lt;/PositionToleranceIncreaseWithSlope&gt;
    ///  &lt;MaxTrials&gt;2&lt;/MaxTrials&gt;
    ///  &lt;BaseThickness&gt;210&lt;/BaseThickness&gt;
    ///  &lt;SelectionFunction&gt;(DST / 0.005)^2 + (DSL / (0.005 + PSL * 0.05))^2 + (DPT/20)^2+(DPL/(20+6*PSL))^2&lt;/SelectionFunction&gt;
    ///  &lt;SelectionFunctionMin&gt;0&lt;/SelectionFunctionMin&gt;
    ///  &lt;SelectionFunctionMax&gt;18&lt;/SelectionFunctionMax&gt;  &lt;RecalibrationSelectionText&gt;select POSX, POSY, SLOPEX, SLOPEY from (select ID_ZONE, POSX, POSY, SLOPEX, SLOPEY, ROW_NUMBER() OVER (PARTITION BY ID_ZONE ORDER
    ///  BY ID) AS RNUM from TB_MIPBASETRACKS where ID_EVENTBRICK = _BRICK_ and ID_ZONE in (select ID from TB_ZONES where ID_EVENTBRICK = _BRICK_ and ID_PROCESSOPERATION
    ///  in (select CALIBRATION from VW_PLATES where ID_EVENTBRICK = _BRICK_ and ID = _PLATE_)) and GRAINS &amp;gt; 27 and SQRT(SLOPEX*SLOPEX+SLOPEY*SLOPEY) &amp;lt; 
    ///  0.25 and SQRT(SLOPEX*SLOPEX+SLOPEY*SLOPEY) &amp;gt; 0.05) where RNUM &amp;lt;= 5&lt;/RecalibrationSelectionText&gt;
    ///  &lt;RecalibrationMinXDistance&gt;30000&lt;/RecalibrationMinXDistance&gt;
    ///  &lt;RecalibrationMinYDistance&gt;30000&lt;/RecalibrationMinYDistance&gt;
    ///  &lt;RecalibrationMinTracks&gt;8&lt;/RecalibrationMinTracks&gt;
    ///  &lt;RecalibrationPosTolerance&gt;60&lt;/RecalibrationPosTolerance&gt;
    ///  &lt;ReadTolerancesFromPredictionTable&gt;False&lt;/ReadTolerancesFromPredictionTable&gt;
    ///  &lt;EnableForking&gt;False&lt;/EnableForking&gt;
    /// &lt;/PredictionScan3Settings&gt;	
    /// </code>
    /// </example>
    /// </para>
    /// <para><b>NOTICE: If the quality cut id is identical to the linker id, no quality cut is applied (unless the linker applies its own quality cuts).</b></para>
    /// <para>A dump file is generated in the <c>Scratch</c> directory, with the name <c>predictionscan3driver_<i>operationid</i>_<i>brick</i>_<i>plate</i>.txt</c> and containing an n-tuple with the following fields:
    /// <list type="table">
    /// <item><term>BRICK</term><description>The id of the brick.</description></item>
    /// <item><term>PLATE</term><description>The id of the plate.</description></item>
    /// <item><term>PATH</term><description>The path number.</description></item>
    /// <item><term>TRIALS</term><description>The trials <b>remaining</b> (not the trials done).</description></item>
    /// <item><term>GRAINS</term><description>The number of grains of the candidate (or zero if no candidate is found).</description></item>
    /// <item><term>SIGMA</term><description>The <c>Sigma</c> of the candidate (negative for microtracks, positive for base tracks, zero for no candidate).</description></item>
    /// </list>
    /// </para>
    /// </remarks>	
    public class Exe : MarshalByRefObject, IInterruptNotifier, SySal.Web.IWebApplication
    {
        /// <summary>
        /// Initializes the Lifetime Service.
        /// </summary>
        /// <returns>the lifetime service object or null.</returns>
        public override object InitializeLifetimeService()
        {
            return null;
        }

        const int ScanFailed = -5;

        static void ShowExplanation()
        {
            ExplanationForm EF = new ExplanationForm();
            System.IO.StringWriter strw = new System.IO.StringWriter();
            strw.WriteLine("");
            strw.WriteLine("PredictionScan3Driver");
            strw.WriteLine("--------------");
            strw.WriteLine("PredictionScan3Driver searches for predicted tracks.");
            strw.WriteLine("All input and output live in the DB. Predictions are read from TB_SCANBACK_PREDICTIONS and results are written to the same table.");
            strw.WriteLine("If needed, forking can be activated to provide multiple results for a single prediction search.");
            strw.WriteLine("Before real scanning, a recalibration can be computed using \"safe\" tracks.");
            strw.WriteLine("The selection must return the following columns, in this order:");
            strw.WriteLine("POSX\tPOSY\tSLOPEX\tSLOPEY");
            strw.WriteLine("_BRICK_ will be replaced with the brick ID");
            strw.WriteLine("_PLATE_ will be replaced with the plate ID");
            strw.WriteLine();
            strw.WriteLine("Type: PredictionScan3Driver /Interrupt <batchmanager> <process operation id> <interrupt string>");
            strw.WriteLine("to send an interrupt message to a running PredictionScan3Driver process operation.");
            strw.WriteLine("SUPPORTED INTERRUPTS:");
            strw.WriteLine("IgnoreScanFailure False|True - instructs PredictionScan3Driver to stop on failed predictions or to skip them and go on.");
            strw.WriteLine("IgnoreRecalFailure False|True - instructs PredictionScan3Driver to stop on failed recalibration tracks or to skip them and go on.");
            strw.WriteLine("Type: PredictionScan3Driver /EasyInterrupt for a graphical user interface to send interrupts.");
            strw.WriteLine("Type: PredictionScan3Driver /CSBrick <brick> [<candidate file>] for a graphical user interface to choose candidate tracks in the brick to connect to CS or TT predictions, or try to guess vertices.");
            strw.WriteLine("--------------");
            strw.WriteLine("The following substitutions apply (case is disregarded):");
            strw.WriteLine("%EXEREP% = Executable repository path specified in the Startup file.");
            strw.WriteLine("%RWDDIR% = Output directory for Raw Data.");
            strw.WriteLine("%TLGDIR% = Output directory for linked zones.");
            strw.WriteLine("%RWD% = Scanning output file name (not including extension).");
            strw.WriteLine("%TLG% = Linked zone file name (not including extension).");
            strw.WriteLine("%SCRATCH% = Scratch directory specified in the Startup file.");
            strw.WriteLine("%ZONEID% = Hexadecimal file name for a zone.");
            strw.WriteLine("--------------");
            strw.WriteLine("The program settings should have the following structure:");
            PredictionScan3Settings pset = new PredictionScan3Settings();
            pset.LinkConfigId = 108888238;
            pset.QualityCutId = 108382880;
            pset.ScanningConfigId = 105382855;
            pset.MaxTrials = 2;
            pset.PositionTolerance = 20.0;
            pset.PositionToleranceIncreaseWithSlope = 6.0;
            pset.SlopeTolerance = 0.03;
            pset.SlopeToleranceIncreaseWithSlope = 0.3;
            pset.SelectionFunction = "(DST / 0.003)^2 + (DSL / (0.003 + PSL * 0.3))^2";
            pset.SelectionFunctionMin = 0.0;
            pset.SelectionFunctionMax = 9.0;
            pset.BaseThickness = 210.0;
            pset.RecalibrationSelectionText = "select POSX, POSY, SLOPEX, SLOPEY from (select ID_ZONE, POSX, POSY, SLOPEX, SLOPEY, ROW_NUMBER() OVER (PARTITION BY ID_ZONE ORDER BY ID) AS RNUM from " +
                "TB_MIPBASETRACKS where ID_EVENTBRICK = _BRICK_ and ID_ZONE in (select ID from TB_ZONES where ID_EVENTBRICK = _BRICK_ and ID_PROCESSOPERATION in " +
                "(select CALIBRATION from VW_PLATES where ID_EVENTBRICK = _BRICK_ and ID = _PLATE_)) and GRAINS > 27 and " +
                "SQRT(SLOPEX*SLOPEX+SLOPEY*SLOPEY) < 0.25 and SQRT(SLOPEX*SLOPEX+SLOPEY*SLOPEY) > 0.05) where RNUM <= 5";
            pset.ReadTolerancesFromPredictionTable = false;
            pset.RecalibrationMinTracks = 8;
            pset.RecalibrationMinXDistance = 30000;
            pset.RecalibrationMinYDistance = 30000;
            pset.RecalibrationPosTolerance = 60;
            pset.LocalIntercalibration = IntercalibrationMode.None;
            new System.Xml.Serialization.XmlSerializer(typeof(PredictionScan3Settings)).Serialize(strw, pset);
            strw.WriteLine("");
            strw.WriteLine("");
            strw.WriteLine("NOTICE: If the quality cut id is identical to the linker id, no quality cut is applied (unless the linker applies its own quality cuts).");
            strw.WriteLine("--------------");
            strw.WriteLine(SyntaxHelp());
            EF.RTFOut.Text = strw.ToString();
            EF.ShowDialog();
        }
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        internal static void Main(string[] args)
        {
            HE = SySal.DAQSystem.Drivers.HostEnv.Own;
            if (HE == null)            
            {
                if (args.Length == 4 && String.Compare(args[0].Trim(), "/Interrupt", true) == 0) SendInterrupt(args[1], Convert.ToInt64(args[2]), args[3]);
                else if (args.Length == 1 && String.Compare(args[0].Trim(), "/EasyInterrupt", true) == 0) EasyInterrupt();
                else if ((args.Length == 3 || args.Length == 2) && String.Compare(args[0].Trim(), "/CSBrick", true) == 0)
                {
                    System.Windows.Forms.Application.EnableVisualStyles();
                    new CSToBrickForm(Convert.ToInt32(args[1]), (args.Length >= 3) ? args[2] : null).ShowDialog();
                }
                else ShowExplanation();
                return;
            }

            Execute();
        }

        private static void SendInterrupt(string machine, long op, string interruptdata)
        {
            SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord(); 
            SySal.OperaDb.OperaDbConnection conn = cred.Connect();
            conn.Open();
            string addr = new SySal.OperaDb.OperaDbCommand("SELECT ADDRESS FROM TB_MACHINES WHERE ID = '" + machine + "' OR UPPER(NAME) = '" + machine.ToUpper() + "' OR UPPER(ADDRESS) = '" + machine.ToUpper() + "' AND ISBATCHSERVER <> 0", conn).ExecuteScalar().ToString();
            Console.WriteLine("Contacting Batch Manager " + addr);
            ((SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + addr + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer) + "/BatchManager.rem")).Interrupt(op, cred.OPERAUserName, cred.OPERAPassword, interruptdata);
        }

        private static System.IO.StreamWriter DumpStream = null;

        private static SySal.DAQSystem.Drivers.HostEnv HE = null;

        private static SySal.OperaDb.OperaDbConnection Conn = null;

        private static SySal.OperaDb.OperaDbTransaction Trans = null;

        private static PredictionScan3Settings ProgSettings;

        private static string QualityCut;

        private static string LinkConfig;

        private static object QualityCutExe;

        private static object LinkerExe;

        private static string LIQualityCut;

        private static string LILinkConfig;

        private static object LIQualityCutExe;

        private static object LILinkerExe;

        private static SySal.DAQSystem.Drivers.ScanningStartupInfo StartupInfo;

        private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = null;

        private static SySal.DAQSystem.ScanServer ScanSrv;

        private static long[] ForkBaseId = new long[1] { 0 };

        private static long NextForkId
        {
            get
            {
                lock (ForkBaseId)
                {
                    if (ForkBaseId[0] == 0)
                        lock (Conn)
                        {
                            return ForkBaseId[0] = 1 + SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("select nvl(max(path),0) from tb_scanback_paths where id_eventbrick = " + StartupInfo.Plate.BrickId + " and id_processoperation = (select id_parent_operation from tb_proc_operations where id = " + StartupInfo.ProcessOperationId + ")", Conn, null).ExecuteScalar());
                        }
                    else
                        return ++ForkBaseId[0];
                }
            }
        }

        private static int TotalZones = 0;

        private static SySal.BasicTypes.Vector2 PredictionsCenter = new SySal.BasicTypes.Vector2();

        private static SySal.BasicTypes.Vector2 PredictionsSlopeAdjust = new SySal.BasicTypes.Vector2();

        private static SySal.DAQSystem.Scanning.IntercalibrationInfo DirRecal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();

        private static SySal.DAQSystem.Scanning.IntercalibrationInfo InvRecal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();

        private static bool RecalibrationDone = false;

        private static int RecalibrationTracksToScan = 0;

        private static char UsedMarkSet;

        private static double PlateMinX = 0.0, PlateMinY = 0.0, PlateMaxX = 0.0, PlateMaxY = 0.0;

        private static System.Threading.ManualResetEvent ProcessEvent = new System.Threading.ManualResetEvent(true);

        private static System.Threading.AutoResetEvent RecalEvent = new System.Threading.AutoResetEvent(false);

        private static System.Collections.ArrayList ScanQueue = new System.Collections.ArrayList();

        private static System.Threading.Thread ThisThread = null;

        private static System.Threading.Thread DBKeepAliveThread = null;

        private static void DBKeepAliveThreadExec()
        {
            try
            {
                SySal.OperaDb.OperaDbCommand keepalivecmd = null;
                lock (Conn)
                    keepalivecmd = new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM DUAL", Conn);
                while (Conn != null)
                {
                    keepalivecmd.ExecuteScalar();
                    System.Threading.Thread.Sleep(10000);
                }
            }
            catch (System.Threading.ThreadAbortException)
            {
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception) { }
        }

        private static System.Threading.Thread WorkerThread = new System.Threading.Thread(new System.Threading.ThreadStart(WorkerThreadExec));

        private static System.Collections.Queue WorkQueue = new System.Collections.Queue();

        private static void WorkerThreadExec()
        {
            try
            {
                while (true)
                {
                    int qc;
                    try
                    {
                        System.Threading.Thread.Sleep(System.Threading.Timeout.Infinite);
                    }
                    catch (System.Threading.ThreadInterruptedException) { }
                    lock (WorkQueue)
                        if ((qc = WorkQueue.Count) == 0) return;
                    while (qc > 0)
                    {
                        PostProcess();
                        lock (WorkQueue) qc = WorkQueue.Count;
                    }
                }
            }
            catch (System.Threading.ThreadAbortException)
            {
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception) { }
        }

        private static System.Exception ThisException = null;

        private static bool IgnoreScanFailure = false;

        private static bool IgnoreRecalFailure = true;

        private static System.Collections.ArrayList ScanningResults = new System.Collections.ArrayList();

        private static IntercalTrack[] IntercalTracks = null;

        private static double RefNominalZ = 0.0;

        private static double CurrNominalZ = 0.0;

        private static double RefCalibZ = 0.0;

        private static double PredictionCorrectDeltaZ = 0.0;

        private static System.Collections.ArrayList RecalibrationResults = new System.Collections.ArrayList();

        private static System.Drawing.Bitmap gIm = null;

        private static System.Drawing.Graphics gMon = null;

        private static NumericalTools.Plot gPlot = null;

        private static bool IsFirstPlot = true;

        private static NumericalTools.CStyleParsedFunction SelFunc = null;

        private static void EasyInterrupt()
        {
            (new frmEasyInterrupt()).ShowDialog();
        }

        private delegate void dPostProcess();

        private static void PostProcess()
        {
            try
            {
                while (true)
                {
                    Prediction zd = null;
                    object o = null;
                    lock (ScanQueue)
                    {
                        if (WorkQueue.Count == 0)
                        {
                            ThisException = null;
                            ProcessEvent.Set();
                            break;
                        }
                        o = WorkQueue.Dequeue();
                        if (o is ManualCheck)
                        {
                            WriteManualCheckResult((ManualCheck)o);
                            continue;
                        }
                    }
                    zd = (Prediction)o;
                    long xid = 0;
                    string[] parameters = SelFunc.ParameterList;
                    int p;

                    string corrfile = "";
                    if (System.IO.File.Exists(StartupInfo.ScratchDir + @"\fragmentshiftcorrection_" + StartupInfo.MachineId + ".xml"))
                        corrfile = " " + StartupInfo.ScratchDir + @"\fragmentshiftcorrection_" + StartupInfo.MachineId + ".xml";

                    System.Globalization.CultureInfo InvC = System.Globalization.CultureInfo.InvariantCulture;

                    SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = null;

                    xid = zd.Series;
                    ScanningResult res = new ScanningResult();
                    res.ForkPaths = new System.Collections.ArrayList();
                    res.PathId = zd.Series;
                    res.CandidateId = 0;
                    res.PPX = zd.PredictedPosX;
                    res.PPY = zd.PredictedPosY;
                    res.PSX = zd.PredictedSlopeX;
                    res.PSY = zd.PredictedSlopeY;
                    res.DPX = 0.0;
                    res.DPY = 0.0;
                    res.DSX = 0.0;
                    res.DSY = 0.0;
                    HE.Write("Processing " + zd.Series + ": ...");

                    if (zd.CandidateIndex == ScanFailed)
                    {
                        res.CandidateId = zd.CandidateIndex;
                        zd.MaxTrials = 0;
                        if (zd.Series > 0)
                        {
                            HE.WriteLine(" Scanning failed");
                            lock (ScanQueue)
                            {
                                ScanningResults.Add(res);
                                WriteResult(zd, res, null);
                            }
                        }
                        else
                        {
                            HE.WriteLine(" Scanning failed");
                            lock (ScanQueue)
                            {
                                RecalibrationResults.Add(res);
                                RecalibrationTracksToScan--;
                                if (RecalibrationTracksToScan == 0)
                                {
                                    if (!RecalibrationDone) ComputeRecalibration();
                                    ThisException = null;
                                    RecalEvent.Set();
                                }
                            }
                        }
                    }
                    else
                    {
/*
                        RecalibrationResults.Add(res);
                        RecalibrationTracksToScan--;
                        */
                        lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)(LinkerExe.GetType().InvokeMember("ProcessData", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, LinkerExe, new object[4] { ReplaceStrings(zd.Outname + ".rwc", zd.Series, ""), null, (zd.Series >= 0 || ProgSettings.LocalIntercalibration == IntercalibrationMode.None) ? LinkConfig : LILinkConfig, null }));
                        if (QualityCutExe != null) lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)(QualityCutExe.GetType().InvokeMember("ProcessData", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.InvokeMethod | System.Reflection.BindingFlags.Instance, null, QualityCutExe, new object[3] { lz, (zd.Series >= 0 || ProgSettings.LocalIntercalibration == IntercalibrationMode.None) ? QualityCut : LIQualityCut, false }));
                        if (RecalibrationDone) lz = new LinkedZone(lz);
                        SySal.OperaPersistence.Persist(ReplaceStrings(zd.Outname + ".tlg", zd.Series, ""), lz);

                        if (zd.Series < 0 && ProgSettings.LocalIntercalibration != IntercalibrationMode.None)
                        {
                            DirRecal.MXX = InvRecal.MXX = 1.0;
                            DirRecal.MYY = InvRecal.MYY = 1.0;
                            DirRecal.MXY = InvRecal.MXY = 0.0;
                            DirRecal.MYX = InvRecal.MYX = 0.0;
                            if (IntercalTracks.Length > 0)
                            {
                                SySal.Processing.QuickMapping.QuickMapper QM = new SySal.Processing.QuickMapping.QuickMapper();
                                SySal.Processing.QuickMapping.Configuration QMC = (SySal.Processing.QuickMapping.Configuration)QM.Config;
                                QMC.FullStatistics = true;
                                QMC.UseAbsoluteReference = false;
                                QMC.PosTol = ProgSettings.RecalibrationPosTolerance;
                                QMC.SlopeTol = ProgSettings.LocalIntercalibrationSlopeTolerance;
                                QM.Config = QMC;
                                SySal.Tracking.MIPEmulsionTrackInfo[] intercalref = new SySal.Tracking.MIPEmulsionTrackInfo[IntercalTracks.Length];
                                SySal.Tracking.MIPEmulsionTrackInfo[] newref = new SySal.Tracking.MIPEmulsionTrackInfo[lz.Length];
                                int t;
                                for (t = 0; t < intercalref.Length; t++) intercalref[t] = IntercalTracks[t].Info;
                                for (t = 0; t < newref.Length; t++) newref[t] = lz[t].Info;
                                SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] pairs = new SySal.Scanning.PostProcessing.PatternMatching.TrackPair[0];
                                double bestdeltaz = 0.0;
                                System.Collections.ArrayList intercalrefids = new System.Collections.ArrayList(intercalref.Length);
                                for (t = 0; t < intercalref.Length; t++) intercalrefids.Add(t);

                                if (ProgSettings.LocalExcludedDeltaZ != null && ProgSettings.LocalExcludedDeltaZ.Length > 0)
                                {
                                    foreach (double deltaz in ProgSettings.LocalExcludedDeltaZ)
                                    {
                                        SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] xclpairs = QM.Match(intercalref, newref, deltaz + RefNominalZ - CurrNominalZ, ProgSettings.RecalibrationMinXDistance, ProgSettings.RecalibrationMinYDistance);
                                        if (xclpairs.Length >= ProgSettings.RecalibrationMinTracks)
                                            foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair xp in xclpairs)
                                                intercalrefids[xp.First.Index] = -1;
                                    }
                                    int totalsurvivors = 0;
                                    for (t = 0; t < intercalrefids.Count; t++)
                                        if ((int)intercalrefids[t] >= 0)
                                            totalsurvivors++;
                                    if (totalsurvivors < intercalref.Length)
                                    {
                                        SySal.Tracking.MIPEmulsionTrackInfo[] tempref = new SySal.Tracking.MIPEmulsionTrackInfo[totalsurvivors];
                                        totalsurvivors = 0;
                                        for (t = 0; t < intercalref.Length; t++)
                                            if ((int)intercalrefids[t] >= 0)
                                                tempref[totalsurvivors++] = intercalref[t];
                                        for (t = 0; t < intercalrefids.Count; t++)
                                            if ((int)intercalrefids[t] < 0)
                                                intercalrefids.RemoveAt(t--);
                                        intercalref = tempref;
                                    }
                                }


                                foreach (double deltaz in ProgSettings.LocalDeltaZ)
                                {
                                    SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] trialpairs = QM.Match(intercalref, newref, deltaz, ProgSettings.RecalibrationMinXDistance, ProgSettings.RecalibrationMinYDistance);
                                    if (trialpairs.Length > pairs.Length)
                                    {
                                        pairs = trialpairs;
                                        bestdeltaz = deltaz;
                                    }
                                }
                                if (pairs.Length < ProgSettings.RecalibrationMinTracks) throw new Exception("Too few tracks found matching in local intercalibration: " + ProgSettings.RecalibrationMinTracks + " required, " + pairs.Length + " found!");

                                HE.WriteLine("base tracks = " + lz.Length + ", pairs = " + pairs.Length + ", DeltaZ = " + bestdeltaz.ToString("F1"));

                                if (ProgSettings.LocalIntercalibration == IntercalibrationMode.RotoTranslationDeltaZ)
                                {
                                    double[] x = new double[pairs.Length];
                                    double[] y = new double[pairs.Length];
                                    double[] dx = new double[pairs.Length];
                                    double[] dy = new double[pairs.Length];
                                    double[] sx = new double[pairs.Length];
                                    double[] sy = new double[pairs.Length];
                                    int g;
                                    for (g = 0; g < pairs.Length; g++)
                                    {
                                        x[g] = pairs[g].Second.Info.Intercept.X - PredictionsCenter.X;
                                        y[g] = pairs[g].Second.Info.Intercept.Y - PredictionsCenter.Y;
                                        dx[g] = pairs[g].Second.Info.Intercept.X - pairs[g].First.Info.Intercept.X;
                                        dy[g] = pairs[g].Second.Info.Intercept.Y - pairs[g].First.Info.Intercept.Y;
                                        sx[g] = pairs[g].First.Info.Slope.X;
                                        sy[g] = pairs[g].First.Info.Slope.Y;
                                        PredictionsSlopeAdjust.X += pairs[g].Second.Info.Slope.X - pairs[g].First.Info.Slope.X;
                                        PredictionsSlopeAdjust.Y += pairs[g].Second.Info.Slope.Y - pairs[g].First.Info.Slope.Y;
                                    }

                                    PredictionsSlopeAdjust.X /= pairs.Length;
                                    PredictionsSlopeAdjust.Y /= pairs.Length;
                                    HE.WriteLine("Prediction slope adjustment (X,Y): " + PredictionsSlopeAdjust.X.ToString("F4") + " " + PredictionsSlopeAdjust.Y.ToString("F4"));

                                    double a = 0.0, corrx = 0.0, corry = 0.0, finedz = 0.0, rot = 0.0, tx = 0.0, ty = 0.0;
                                    if (NumericalTools.Fitting.LinearFitSE(sx, dx, ref corrx, ref a, ref a, ref a, ref a, ref a, ref a) != NumericalTools.ComputationResult.OK) throw new Exception("DeltaZ X computation failed. " + pairs.Length + " tracks used.");
                                    if (NumericalTools.Fitting.LinearFitSE(sy, dy, ref corry, ref a, ref a, ref a, ref a, ref a, ref a) != NumericalTools.ComputationResult.OK) throw new Exception("DeltaZ Y computation failed. " + pairs.Length + " tracks used.");
                                    finedz = 0.5 * (corrx + corry);
                                    bestdeltaz = bestdeltaz + finedz;
                                    HE.WriteLine("FineDZ = " + finedz.ToString("F1") + " TotalDZ = " + bestdeltaz.ToString("F1"));

                                    for (g = 0; g < pairs.Length; g++)
                                    {
                                        dx[g] -= sx[g] * finedz;
                                        dy[g] -= sy[g] * finedz;
                                    }

                                    if (NumericalTools.Fitting.LinearFitSE(x, dy, ref corrx, ref a, ref a, ref a, ref a, ref a, ref a) != NumericalTools.ComputationResult.OK) throw new Exception("Rotation X computation failed. " + pairs.Length + " tracks used.");
                                    if (NumericalTools.Fitting.LinearFitSE(y, dx, ref corry, ref a, ref a, ref a, ref a, ref a, ref a) != NumericalTools.ComputationResult.OK) throw new Exception("Rotation Y computation failed. " + pairs.Length + " tracks used.");

                                    rot = 0.5 * (corrx - corry);
                                    HE.WriteLine("Rotation = " + rot.ToString("F5"));
                                    for (g = 0; g < pairs.Length; g++)
                                    {
                                        tx += dx[g] + y[g] * rot;
                                        ty += dy[g] - x[g] * rot;
                                    }
                                    tx /= pairs.Length;
                                    ty /= pairs.Length;

                                    HE.WriteLine("Translations (X,Y) = " + tx.ToString("F1") + " " + ty.ToString("F1"));

                                    DirRecal.MXX = DirRecal.MYY = Math.Sqrt(1.0 - rot * rot);
                                    DirRecal.MXY = -rot;
                                    DirRecal.MYX = rot;
                                    DirRecal.TX = tx + (DirRecal.MXX - 1.0) * (DirRecal.RX - PredictionsCenter.X) + DirRecal.MXY * (DirRecal.RY - PredictionsCenter.Y);
                                    DirRecal.TY = ty + DirRecal.MYX * (DirRecal.RX - PredictionsCenter.X) + (DirRecal.MYY - 1.0) * (DirRecal.RY - PredictionsCenter.Y);
                                    ComputeInverse();

                                    MyTransformation.Transformation = InvRecal;
                                    lz = new LinkedZone(lz);
                                    long newid = SySal.OperaDb.Scanning.LinkedZone.Save(lz, StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, 0, zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1), System.IO.File.GetCreationTime(zd.Outname + ".rwc"), System.IO.File.GetCreationTime(zd.Outname + ".tlg"), Conn, Trans);                                    
                                    if (Conn.HasBufferTables)
                                    {
                                        foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair pair in pairs)
                                            SySal.OperaDb.Schema.LZ_PATTERN_MATCH.Insert(StartupInfo.Plate.BrickId, StartupInfo.ProcessOperationId, IntercalTracks[(int)intercalrefids[pair.First.Index]].IdZone, newid, IntercalTracks[(int)intercalrefids[pair.First.Index]].Id, pair.Second.Index + 1);
                                        SySal.OperaDb.Schema.LZ_PATTERN_MATCH.Flush();
                                    }
                                    else
                                    {
                                        foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair pair in pairs)
                                            SySal.OperaDb.Schema.TB_PATTERN_MATCH.Insert(StartupInfo.Plate.BrickId, IntercalTracks[(int)intercalrefids[pair.First.Index]].IdZone, newid, IntercalTracks[(int)intercalrefids[pair.First.Index]].Id, pair.Second.Index + 1, StartupInfo.ProcessOperationId);
                                        SySal.OperaDb.Schema.TB_PATTERN_MATCH.Flush();
                                    }

                                    double z = bestdeltaz + CurrNominalZ + RefCalibZ - RefNominalZ;
                                    PredictionCorrectDeltaZ = (z - CurrNominalZ) - (RefCalibZ - RefNominalZ);
                                    new SySal.OperaDb.OperaDbCommand("CALL PC_CALIBRATE_PLATE(" + StartupInfo.Plate.BrickId + ", " + StartupInfo.Plate.PlateId + ", " + StartupInfo.ProcessOperationId + ", '" + UsedMarkSet + "', " + z.ToString(InvC) + ", " +
                                        InvRecal.MXX.ToString(InvC) + ", " + InvRecal.MXY.ToString(InvC) + ", " + InvRecal.MYX.ToString(InvC) + ", " + InvRecal.MYY.ToString(InvC) +
                                        ", " + InvRecal.TX.ToString(InvC) + ", " + InvRecal.TY.ToString(InvC) + ")", Conn, Trans).ExecuteNonQuery();

                                    RecalibrationDone = true;
                                    RecalEvent.Set();
                                }
                                else throw new Exception("Intercalibration mode " + ProgSettings.LocalIntercalibration + " not supported!");
                            }
                            else
                            {
                                MyTransformation.Transformation = InvRecal;
                                lz = new LinkedZone(lz);
                                SySal.OperaDb.Scanning.LinkedZone.Save(lz, StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, 0, zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1), System.IO.File.GetCreationTime(zd.Outname + ".rwc"), System.IO.File.GetCreationTime(zd.Outname + ".tlg"), Conn, Trans);

                                HE.WriteLine("base tracks: " + lz.Length);
                                double z = Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, Conn, Trans).ExecuteScalar());
                                new SySal.OperaDb.OperaDbCommand("CALL PC_CALIBRATE_PLATE(" + StartupInfo.Plate.BrickId + ", " + StartupInfo.Plate.PlateId + ", " + StartupInfo.ProcessOperationId + ", '" + UsedMarkSet + "', " + z.ToString(InvC) + ", " +
                                    DirRecal.MXX.ToString(InvC) + ", " + DirRecal.MXY.ToString(InvC) + ", " + DirRecal.MYX.ToString(InvC) + ", " + DirRecal.MYY.ToString(InvC) +
                                    ", " + DirRecal.TX.ToString(InvC) + ", " + DirRecal.TY.ToString(InvC) + ")", Conn, Trans).ExecuteNonQuery();

                                RecalibrationDone = true;
                                RecalEvent.Set();
                            }
                        }
                        else
                        {
                            int t = 0;
                            zd.CandidateIndex = 0;
                            zd.CandidateInfo = null;
                            double nx, ny;
                            double slope = Math.Sqrt(zd.PredictedSlopeX * zd.PredictedSlopeX + zd.PredictedSlopeY * zd.PredictedSlopeY);
                            if (zd.ToleranceFrame == SySal.DAQSystem.Frame.Cartesian)
                            {
                                nx = 0.0;
                                ny = 1.0;
                            }
                            else
                            {
                                if (slope > 0.0)
                                {
                                    nx = zd.PredictedSlopeX / slope;
                                    ny = zd.PredictedSlopeY / slope;
                                }
                                else
                                {
                                    nx = 1.0;
                                    ny = 0.0;
                                }
                            }
                            double selbestval = 2.0 * Math.Max(Math.Abs(ProgSettings.SelectionFunctionMax), Math.Abs(ProgSettings.SelectionFunctionMin)) + 1.0;
                            double selval = 0.0;
                            double dpx = 0.0, dpy = 0.0, dsx = 0.0, dsy = 0.0, dpc, dp = 0.0;
                            zd.CandidateIndex = 0;
                            zd.CandidateInfo = null;
                            for (t = 0; t < lz.Length; t++)
                            {
                                SySal.Tracking.MIPEmulsionTrackInfo info = lz[t].Info;
                                dsx = info.Slope.X - zd.PredictedSlopeX - PredictionsSlopeAdjust.X;
                                dsy = info.Slope.Y - zd.PredictedSlopeY - PredictionsSlopeAdjust.Y;
                                if (Math.Abs(dsx * ny - dsy * nx) > zd.SlopeTolerance1) continue;
                                if (Math.Abs(dsx * nx + dsy * ny) > zd.SlopeTolerance2) continue;
                                dpx = info.Intercept.X - zd.PredictedPosX;
                                dpy = info.Intercept.Y - zd.PredictedPosY;
                                dpc = (dpx * nx + dpy * ny) / zd.PositionTolerance2;
                                if (Math.Abs(dpc) > 1.0) continue;
                                dp = dpc * dpc;
                                dpc = (dpx * ny - dpy * nx) / zd.PositionTolerance1;
                                if (Math.Abs(dpc) > 1.0) continue;
                                dp += dpc * dpc;

                                for (p = 0; p < parameters.Length; p++)
                                {
                                    string s = parameters[p];
                                    if (String.Compare(s, "PLATE", true) == 0) SelFunc[p] = (double)StartupInfo.Plate.PlateId;
                                    else if (String.Compare(s, "N", true) == 0) SelFunc[p] = info.Count;
                                    else if (String.Compare(s, "A", true) == 0) SelFunc[p] = info.AreaSum;
                                    else if (String.Compare(s, "S", true) == 0) SelFunc[p] = info.Sigma;
                                    else if (String.Compare(s, "PPX", true) == 0) SelFunc[p] = zd.PredictedPosX;
                                    else if (String.Compare(s, "PPY", true) == 0) SelFunc[p] = zd.PredictedPosY;
                                    else if (String.Compare(s, "PSX", true) == 0) SelFunc[p] = zd.PredictedSlopeX;
                                    else if (String.Compare(s, "PSY", true) == 0) SelFunc[p] = zd.PredictedSlopeY;
                                    else if (String.Compare(s, "PSL", true) == 0) SelFunc[p] = slope;
                                    else if (String.Compare(s, "FPX", true) == 0) SelFunc[p] = info.Intercept.X;
                                    else if (String.Compare(s, "FPY", true) == 0) SelFunc[p] = info.Intercept.Y;
                                    else if (String.Compare(s, "FSX", true) == 0) SelFunc[p] = info.Slope.X;
                                    else if (String.Compare(s, "FSY", true) == 0) SelFunc[p] = info.Slope.Y;
                                    else if (String.Compare(s, "FSL", true) == 0) SelFunc[p] = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
                                    else if (String.Compare(s, "TX", true) == 0) SelFunc[p] = nx;
                                    else if (String.Compare(s, "TY", true) == 0) SelFunc[p] = ny;
                                    else if (String.Compare(s, "LX", true) == 0) SelFunc[p] = ny;
                                    else if (String.Compare(s, "LY", true) == 0) SelFunc[p] = -nx;
                                    else if (String.Compare(s, "DPX", true) == 0) SelFunc[p] = dpx;
                                    else if (String.Compare(s, "DPY", true) == 0) SelFunc[p] = dpy;
                                    else if (String.Compare(s, "DPT", true) == 0) SelFunc[p] = dpx * ny - dpy * nx;
                                    else if (String.Compare(s, "DPL", true) == 0) SelFunc[p] = dpx * nx + dpy * ny;
                                    else if (String.Compare(s, "DSX", true) == 0) SelFunc[p] = dsx;
                                    else if (String.Compare(s, "DSY", true) == 0) SelFunc[p] = dsy;
                                    else if (String.Compare(s, "DST", true) == 0) SelFunc[p] = dsx * ny - dsy * nx;
                                    else if (String.Compare(s, "DSL", true) == 0) SelFunc[p] = dsx * nx + dsy * ny;
                                    else if (String.Compare(s, "TRIALS", true) == 0) SelFunc[p] = zd.MaxTrials;
                                }
                                selval = SelFunc.Evaluate();
                                if (selval < ProgSettings.SelectionFunctionMin || selval > ProgSettings.SelectionFunctionMax) continue;
                                // if (ProgSettings.EnableForking)
                                {
                                    ForkInfo fi = new ForkInfo();
                                    fi.CandidateId = t + 1;
                                    fi.SelFuncValue = selval;
                                    int insindex = res.ForkPaths.BinarySearch(fi, ForkInfoComparer.TheForkInfoComparer);
                                    if (insindex < 0) insindex = ~insindex;
                                    res.ForkPaths.Insert(insindex, fi);
                                }
                                if (selval < selbestval)
                                {
                                    selbestval = selval;
                                    zd.CandidateIndex = t + 1;
                                    zd.CandidateInfo = info;
                                    res.CandidateId = zd.CandidateIndex;
                                    res.DPX = dpx;
                                    res.DPY = dpy;
                                    res.DSX = dsx;
                                    res.DSY = dsy;
                                }
                            }
                            if (zd.CandidateIndex == 0)
                            {
                                if (--zd.MaxTrials > 0)
                                {
                                    HE.WriteLine(" Not found, trials to go: " + zd.MaxTrials);
                                    string[] rwds = System.IO.Directory.GetFiles(zd.Outname.Substring(0, zd.Outname.LastIndexOf("\\")), zd.Outname.Substring(zd.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
                                    foreach (string rwd in rwds)
                                        System.IO.File.Delete(rwd);
                                    System.IO.File.Delete(zd.Outname + ".rwc");
                                    lock (ScanQueue)
                                    {
                                        if (ScanQueue.Count >= 5)
                                            ScanQueue.Insert(5, zd);
                                        else
                                            ScanQueue.Insert(ScanQueue.Count, zd);
                                    }
                                }
                                else
                                {
                                    if (zd.Series > 0)
                                    {
                                        lock (ScanQueue)
                                        {
                                            ScanningResults.Add(res);
                                            WriteResult(zd, res, lz);
                                        }
                                    }
                                    else
                                    {
                                        HE.WriteLine(" Not found, abandoned.");
                                        lock (ScanQueue)
                                        {
                                            RecalibrationResults.Add(res);
                                            RecalibrationTracksToScan--;
                                            if (RecalibrationTracksToScan == 0)
                                            {
                                                if (!RecalibrationDone) ComputeRecalibration();
                                                ThisException = null;
                                                RecalEvent.Set();
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (zd.Series > 0)
                                {
                                    lock (ScanQueue)
                                    {
                                        ScanningResults.Add(res);
                                        WriteResult(zd, res, lz);
                                    }
                                }
                                else
                                {
                                    HE.WriteLine(" Found.");
                                    lock (ScanQueue)
                                    {
                                        RecalibrationResults.Add(res);
                                        RecalibrationTracksToScan--;
                                        if (RecalibrationTracksToScan == 0)
                                        {
                                            if (!RecalibrationDone) ComputeRecalibration();
                                            ThisException = null;
                                            RecalEvent.Set();
                                        }
                                    }
                                }
                            }
                        }
                        UpdatePlots();
                        xid = 0;
                        lz = null;
                    }
                }
            }
            catch (Exception x)
            {
                try
                {
                    HE.WriteLine("Exception:\r\n" + x.ToString());
                }
                catch (Exception) { }
                ThisException = x;
                ProcessEvent.Set();
                RecalEvent.Set();
            }
        }

        static string ResultText = "";

        private static void WriteResultFile(SySal.DAQSystem.Drivers.Prediction pred, ScanningResult res, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz)
        {
            try
            {
                System.IO.StringWriter resf = new System.IO.StringWriter();
/*                string resfname = StartupInfo.ScratchDir;
                if (resfname.EndsWith("\\") == false && resfname.EndsWith("/") == false) resfname += "\\";
                resfname += ProgSettings.ResultFile + "_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.Plate.PlateId + "_" + StartupInfo.ProcessOperationId + ".txt";
                System.IO.StreamWriter resf = new System.IO.StreamWriter(resfname, true);*/
                foreach (object o in res.ForkPaths)
                    if (o is ForkInfo)
                    {
                        ForkInfo fi = (ForkInfo)o;
                        SySal.Tracking.MIPEmulsionTrackInfo info = lz[(int)fi.CandidateId - 1].Info;
                        resf.WriteLine(StartupInfo.Plate.BrickId + "\t" + StartupInfo.Plate.PlateId + "\t" + pred.Series + " \t" + StartupInfo.ProcessOperationId + "\t" +
                            pred.PredictedPosX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            pred.PredictedPosY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            pred.PredictedSlopeX.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            pred.PredictedSlopeY.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            info.Count + "\t" + info.AreaSum + "\t" +
                            info.Intercept.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            info.Intercept.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            info.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            info.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            info.Sigma.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (pred.PredictedPosX - info.Intercept.X).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (pred.PredictedPosY - info.Intercept.Y).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (pred.PredictedSlopeX - info.Slope.X).ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (pred.PredictedSlopeY - info.Slope.Y).ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    }
                foreach (object o in res.ForkPaths)
                    if (o is ManualCheck)
                    {
                        ManualCheck mc = (ManualCheck)o;
                        resf.WriteLine(StartupInfo.Plate.BrickId + "\t" + StartupInfo.Plate.PlateId + "\t" + mc.Input.Id + " \t" + StartupInfo.ProcessOperationId + "\t" +
                            mc.Input.Position.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Input.Position.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Input.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Input.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Output.Grains + "\t0\t" +
                            mc.Output.Position.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Output.Position.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Output.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            mc.Output.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t0\t" +
                            (mc.Input.Position.X - mc.Output.Position.X).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (mc.Input.Position.Y - mc.Output.Position.Y).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (mc.Input.Slope.X - mc.Output.Slope.X).ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                            (mc.Input.Slope.Y - mc.Output.Slope.Y).ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    }
                resf.Flush();
                ResultText += resf.ToString();
                resf.Close();                
            }
            catch (Exception x)
            {
                HE.WriteLine("Exception:\r\n" + x.Message);
                throw x;
            }
        }

        private static void WriteManualCheckResult(ManualCheck mc)
        {
            if (ProgSettings.ResultFile != null && ProgSettings.ResultFile.Length > 0 && mc.Output.CheckDone && mc.Output.Found)
            {
                ScanningResult res = new ScanningResult();
                res.ForkPaths = new System.Collections.ArrayList(1);
                res.ForkPaths.Add(mc);
                Prediction pred = new Prediction();
                WriteResultFile(null, res, null);
                return;
            }
            try
            {
                if (mc.Output.CheckDone == false || mc.Output.Found == false)
                {
                    SySal.OperaDb.Scanning.Procedures.ScanbackNoCandidate(StartupInfo.Plate.BrickId, (int)StartupInfo.Plate.PlateId, mc.PredictionId, mc.ZoneId, Conn, Trans);
                }
                else
                {
                    if (Conn.HasBufferTables)
                    {
                        SySal.OperaDb.Schema.LZ_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, mc.ZoneId, 1, mc.NextTopMicroId, mc.Output.Position.X, mc.Output.Position.Y, mc.Output.Slope.X, mc.Output.Slope.Y, 0, 0, System.DBNull.Value, 0, 1);
                        SySal.OperaDb.Schema.LZ_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, mc.ZoneId, 2, mc.NextBottomMicroId, mc.Output.Position.X - mc.BaseThickness * mc.Output.Slope.X, mc.Output.Position.Y - mc.BaseThickness * mc.Output.Slope.Y, mc.Output.Slope.X, mc.Output.Slope.Y, 0, 0, System.DBNull.Value, 0, 1);
                        SySal.OperaDb.Schema.LZ_MIPMICROTRACKS.Flush();
                        SySal.OperaDb.Schema.LZ_MIPBASETRACKS.Insert(StartupInfo.Plate.BrickId, mc.ZoneId, mc.NextBaseId, mc.Output.Position.X, mc.Output.Position.Y, mc.Output.Slope.X, mc.Output.Slope.Y, 0, 0, System.DBNull.Value, 0, 1, mc.NextTopMicroId, 2, mc.NextBottomMicroId);
                        SySal.OperaDb.Schema.LZ_MIPBASETRACKS.Flush();
                    }
                    else
                    {
                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, mc.ZoneId, 1, mc.NextTopMicroId, mc.Output.Position.X, mc.Output.Position.Y, mc.Output.Slope.X, mc.Output.Slope.Y, 0, 0, System.DBNull.Value, 0, 1);
                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.Plate.BrickId, mc.ZoneId, 2, mc.NextBottomMicroId, mc.Output.Position.X - mc.BaseThickness * mc.Output.Slope.X, mc.Output.Position.Y - mc.BaseThickness * mc.Output.Slope.Y, mc.Output.Slope.X, mc.Output.Slope.Y, 0, 0, System.DBNull.Value, 0, 1);
                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Flush();
                        SySal.OperaDb.Schema.TB_MIPBASETRACKS.Insert(StartupInfo.Plate.BrickId, mc.ZoneId, mc.NextBaseId, mc.Output.Position.X, mc.Output.Position.Y, mc.Output.Slope.X, mc.Output.Slope.Y, 0, 0, System.DBNull.Value, 0, 1, mc.NextTopMicroId, 2, mc.NextBottomMicroId);
                        SySal.OperaDb.Schema.TB_MIPBASETRACKS.Flush();
                    }
                    SySal.OperaDb.Scanning.Procedures.ScanbackCandidate(StartupInfo.Plate.BrickId, (int)StartupInfo.Plate.PlateId, mc.PredictionId, mc.ZoneId, mc.NextBaseId, true, Conn, Trans);
                }
                HE.WriteLine(" Zone: " + mc.ZoneId + " Manual check result: " + ((mc.Output.Found && mc.Output.CheckDone) ? "Found" : "Not found"));
            }
            catch (Exception x)
            {
                HE.WriteLine("Exception:\r\n" + x.Message);
                throw x;
            }
        }

        private static void WriteResult(SySal.DAQSystem.Drivers.Prediction pred, ScanningResult res, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz)
        {
            if (ProgSettings.ResultFile != null && ProgSettings.ResultFile.Length > 0)
            {
                WriteResultFile(pred, res, lz);
                return;
            }
            try
            {
                if (pred.Series > 0)
                {
                    if (pred.CandidateIndex > 0)
                        DumpStream.Write(StartupInfo.Plate.BrickId + "\t" + StartupInfo.Plate.PlateId + "\t" + pred.Series + "\t" + pred.MaxTrials + "\t" + pred.CandidateInfo.Count + "\t" + pred.CandidateInfo.Sigma.ToString("F3"));
                    else DumpStream.Write(StartupInfo.Plate.BrickId + "\t" + StartupInfo.Plate.PlateId + "\t" + pred.Series + "\t0\t0\t0");
                }
            }
            catch (Exception) { }
            try
            {
                if (res.CandidateId == ScanFailed)
                {
                    //new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DAMAGEDZONE(" + StartupInfo.Plate.BrickId + ", " + StartupInfo.Plate.PlateId + ", " + res.PathId + ", 'U')", Conn, Trans).ExecuteNonQuery();
                    if (ProgSettings.AvoidWritingScanbackPredictions == false) SySal.OperaDb.Scanning.Procedures.ScanbackDamagedZone(StartupInfo.Plate.BrickId, (int)StartupInfo.Plate.PlateId, res.PathId, 'U', Conn, Trans);
                    return;
                }
                if (lz == null)
                {
                    System.IO.FileStream r = new System.IO.FileStream(pred.Outname + ".tlg", System.IO.FileMode.Open, System.IO.FileAccess.Read);
                    lz = new SySal.Scanning.Plate.IO.OPERA.LinkedZone(r);
                    r.Close();
                }
                long zoneid = SySal.OperaDb.Scanning.LinkedZone.Save(lz, StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, pred.Series, pred.Outname.Substring(pred.Outname.LastIndexOf("\\") + 1), System.IO.File.GetCreationTime(pred.Outname + ".rwc"), System.IO.File.GetCreationTime(pred.Outname + ".tlg"), Conn, Trans);
                HE.WriteLine(" Zone: " + zoneid + " Tracks: " + lz.Length + " Id_Path: " + res.PathId + " Candidate: " + res.CandidateId);
                if (ProgSettings.AvoidWritingScanbackPredictions) return;
                if (res.CandidateId > 0)
                    SySal.OperaDb.Scanning.Procedures.ScanbackCandidate(StartupInfo.Plate.BrickId, (int)StartupInfo.Plate.PlateId, res.PathId, zoneid, res.CandidateId, false, Conn, Trans);
                else
                {
                    if (ProgSettings.AskManualScanIfMissing)
                    {
                        ManualCheck mc = new ManualCheck();
                        mc.PredictionId = res.PathId;
                        mc.ZoneId = zoneid;
                        mc.NextBaseId = lz.Length + 1;
                        mc.NextTopMicroId = lz.Top.Length + 1;
                        mc.NextBottomMicroId = lz.Bottom.Length + 1;
                        mc.Input.Id = mc.PredictionId;
                        mc.Input.Position.X = res.PPX;
                        mc.Input.Position.Y = res.PPY;
                        mc.Input.PositionTolerance = Math.Max(pred.PositionTolerance1, pred.PositionTolerance2);
                        mc.Input.Slope.X = res.PSX;
                        mc.Input.Slope.Y = res.PSY;
                        mc.Input.SlopeTolerance = Math.Max(pred.SlopeTolerance1, pred.SlopeTolerance2);
                        lock (ScanQueue)
                            ScanQueue.Add(mc);
                    }
                    else SySal.OperaDb.Scanning.Procedures.ScanbackNoCandidate(StartupInfo.Plate.BrickId, (int)StartupInfo.Plate.PlateId, res.PathId, zoneid, Conn, Trans);
                }
                if (ProgSettings.EnableForking && res.ForkPaths.Count > 1)
                {
                    int ind;
                    for (ind = 1; ind < res.ForkPaths.Count; ind++)
                    {
                        ForkInfo fi = (ForkInfo)(res.ForkPaths[ind]);                        
                        fi.ForkPathId = NextForkId;
                        SySal.OperaDb.Scanning.Procedures.ScanbackFork(StartupInfo.Plate.BrickId, (int)StartupInfo.Plate.PlateId, res.PathId, zoneid, fi.CandidateId, fi.ForkPathId, Conn, Trans);
                    }
                }
            }
            catch (Exception x)
            {
                HE.WriteLine("Exception:\r\n" + x.Message);
                throw x;
            }
        }

        class ForwardComparer : System.Collections.IComparer
        {
            public int Compare(object a, object b)
            {
                double x = ((SySal.DAQSystem.Drivers.Prediction)a).PredictedPosX - ((SySal.DAQSystem.Drivers.Prediction)b).PredictedPosX;
                if (x < 0.0) return -1;
                if (x > 0.0) return 1;
                return 0;
            }
        }

        class BackwardComparer : System.Collections.IComparer
        {
            public int Compare(object a, object b)
            {
                double x = ((SySal.DAQSystem.Drivers.Prediction)a).PredictedPosX - ((SySal.DAQSystem.Drivers.Prediction)b).PredictedPosX;
                if (x > 0.0) return -1;
                if (x < 0.0) return 1;
                return 0;
            }
        }

        static System.Collections.ArrayList OptimizePath(System.Collections.ArrayList p)
        {
            if (p.Count < 3) return p;
            double MinY, MaxY, DY;
            MinY = MaxY = ((Prediction)p[0]).PredictedPosY;
            int i, n;
            for (i = 1; i < p.Count; i++)
            {
                Prediction pred = (Prediction)p[i];
                if (MinY > pred.PredictedPosY) MinY = pred.PredictedPosY;
                else if (MaxY < pred.PredictedPosY) MaxY = pred.PredictedPosY;
            }
            n = Convert.ToInt32(Math.Sqrt(p.Count) * 0.25 + 1);
            DY = (MaxY - MinY) / n;
            if (DY <= 0.0) DY = 1.0;
            System.Collections.ArrayList[] strips = new System.Collections.ArrayList[n + 1];
            for (i = 0; i <= n; i++)
                strips[i] = new System.Collections.ArrayList();
            System.Collections.IComparer fc = new ForwardComparer();
            System.Collections.IComparer bc = new BackwardComparer();
            foreach (SySal.DAQSystem.Drivers.Prediction pi in p)
            {
                i = Convert.ToInt32((pi.PredictedPosY - MinY) / DY);
                System.Collections.ArrayList s = strips[i];
                int index = s.BinarySearch(pi, ((i % 2) == 0) ? fc : bc);
                if (index < 0) index = ~index;
                s.Insert(index, pi);
            }
            System.Collections.ArrayList np = new System.Collections.ArrayList(p.Count);
            foreach (System.Collections.ArrayList s in strips)
                np.AddRange(s);
            return np;
        }

        static double PathLen(System.Collections.ArrayList p)
        {
            if (p.Count < 2) return 0.0;
            double len = 0.0;
            int i;
            for (i = 1; i < p.Count; i++)
            {
                Prediction p1 = (Prediction)p[i];
                Prediction p2 = (Prediction)p[i - 1];
                len += Math.Max(Math.Abs(p1.PredictedPosX - p2.PredictedPosX), Math.Abs(p1.PredictedPosY - p2.PredictedPosY));
            }
            return len;
        }

        private static void Execute()
        {
            //			HE.WriteLine("Waiting for you");
            //			HE.ReadLine();
            ThisThread = System.Threading.Thread.CurrentThread;
            gIm = new System.Drawing.Bitmap(360, 360);
            gMon = System.Drawing.Graphics.FromImage(gIm);
            gPlot = new NumericalTools.Plot();

            StartupInfo = (SySal.DAQSystem.Drivers.ScanningStartupInfo)HE.StartupInfo;
            Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
            Conn.Open();
            SySal.OperaDb.Schema.DB = Conn;
            (DBKeepAliveThread = new System.Threading.Thread(DBKeepAliveThreadExec)).Start();
            ScanSrv = HE.ScanSrv;

            if (new SySal.OperaDb.OperaDbCommand("SELECT DAMAGED FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, Conn, null).ExecuteScalar().ToString() != "N")
                throw new Exception("Plate #" + StartupInfo.Plate.PlateId + ", Brick #" + StartupInfo.Plate.BrickId + " is damaged!");

            System.Xml.Serialization.XmlSerializer xmlp = new System.Xml.Serialization.XmlSerializer(typeof(PredictionScan3Driver.PredictionScan3Settings));
            ProgSettings = (PredictionScan3Driver.PredictionScan3Settings)xmlp.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
            if (ProgSettings.LocalIntercalibration != IntercalibrationMode.None)
            {
                long calibrationid;
                StartupInfo.Plate.MapInitString = SySal.OperaDb.Scanning.Utilities.GetMapString(StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, false,
                    SySal.OperaDb.Scanning.Utilities.CharToMarkType(UsedMarkSet = Convert.ToChar(new SySal.OperaDb.OperaDbCommand("SELECT MARKSET FROM TB_PROGRAMSETTINGS WHERE ID = " + StartupInfo.ProgramSettingsId, Conn).ExecuteScalar())),
                    out calibrationid, Conn, null);
            }
            SelFunc = MakeFunctionAndCheck(ProgSettings.SelectionFunction);
            if (ProgSettings.RecalibrationSelectionText != null)
            {
                ProgSettings.RecalibrationSelectionText = ProgSettings.RecalibrationSelectionText.Replace("&gt;", ">");
                ProgSettings.RecalibrationSelectionText = ProgSettings.RecalibrationSelectionText.Replace("&lt;", "<");
            }
            LinkConfig = (string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.LinkConfigId + ")", Conn, null).ExecuteScalar();
            LinkerExe = System.Activator.CreateInstanceFrom(StartupInfo.ExeRepository + @"\BatchLink.exe", "SySal.Executables.BatchLink.Exe").Unwrap();
            if (ProgSettings.QualityCutId == ProgSettings.LinkConfigId)
            {
                QualityCut = null;
                QualityCutExe = null;
            }
            else
            {
                QualityCut = (string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.QualityCutId + ")", Conn, null).ExecuteScalar();
                if (QualityCut.StartsWith("\"") && QualityCut.EndsWith("\"")) QualityCut = QualityCut.Substring(1, QualityCut.Length - 2);
                QualityCutExe = System.Activator.CreateInstanceFrom(StartupInfo.ExeRepository + @"\TLGSel.exe", "SySal.Executables.TLGSel.Exe").Unwrap();
            }
            if (ProgSettings.LocalIntercalibrationLinkConfigId <= 0) ProgSettings.LocalIntercalibrationLinkConfigId = ProgSettings.LinkConfigId;
            if (ProgSettings.LocalIntercalibrationQualityCutId <= 0) ProgSettings.LocalIntercalibrationQualityCutId = ProgSettings.QualityCutId;

            LILinkConfig = (string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.LocalIntercalibrationLinkConfigId + ")", Conn, null).ExecuteScalar();
            LILinkerExe = System.Activator.CreateInstanceFrom(StartupInfo.ExeRepository + @"\BatchLink.exe", "SySal.Executables.BatchLink.Exe").Unwrap();
            if (ProgSettings.LocalIntercalibrationQualityCutId == ProgSettings.LocalIntercalibrationLinkConfigId)
            {
                LIQualityCut = null;
                LIQualityCutExe = null;
            }
            else
            {
                LIQualityCut = (string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE(ID = " + ProgSettings.LocalIntercalibrationQualityCutId + ")", Conn, null).ExecuteScalar();
                if (LIQualityCut.StartsWith("\"") && QualityCut.EndsWith("\"")) LIQualityCut = LIQualityCut.Substring(1, LIQualityCut.Length - 2);
                LIQualityCutExe = System.Activator.CreateInstanceFrom(StartupInfo.ExeRepository + @"\TLGSel.exe", "SySal.Executables.TLGSel.Exe").Unwrap();
            }

            if (ProgSettings.LocalDeltaZ == null || ProgSettings.LocalDeltaZ.Length == 0) ProgSettings.LocalDeltaZ = new double[1] { 0.0 };

            if (ProgSettings.LocalIntercalibration == IntercalibrationMode.AffineDeltaZ) throw new Exception("Unsupported parameter for IntercalibrationMode: " + ProgSettings.LocalIntercalibration);

            if (ProgSettings.LocalIntercalibration == IntercalibrationMode.RotoTranslationDeltaZ)
            {
                ProgSettings.RecalibrationSelectionText =
                    "select round(posx + (currZ - Z) * slopex, 1) as posx, round(posy + (currZ - Z) * slopey, 1) as posy, slopex, slopey, id_zone, idtrack, id_refcalib, Z, currZ, id_plate from " +
                    "(select Z, id_zone, idtrack, posx, posy, slopex, slopey, id_refcalib, id_plate from " +
                    "(select /*+index_asc(tb_mipbasetracks pk_mipbasetracks) */ idbrick, id_plate, id_zone, id as idtrack, posx, posy, slopex, slopey, id_refcalib from tb_mipbasetracks inner join " +
                    "(select /*+index_asc(tb_zones pk_zones) */ id_eventbrick as idbrick, id_plate, id as idz, id_processoperation as id_refcalib from tb_zones z where (id_eventbrick, id_plate, id_processoperation) in " +
                    "(select id_eventbrick, id as id_plate, calibration as idproc from vw_plates where (id_eventbrick, id) in " +
                    "(select id_eventbrick, id_plate as plate from " +
                    "(select id_eventbrick, id_plate, row_number() over (order by abs(id_plate - " + StartupInfo.Plate.PlateId + ") asc) as rnum from " +
                    "(select distinct id_eventbrick, id_plate from tb_scanback_predictions where (id_eventbrick, id_path) in " +
                    "(select id_eventbrick, id from tb_scanback_paths where (id_eventbrick, id_processoperation) in " +
                    "(select id_eventbrick, id_parent_operation from tb_proc_operations where id = " + StartupInfo.ProcessOperationId + ")) and id_plate <> " + StartupInfo.Plate.PlateId + ")) where rnum = 1) )  and not exists (select * from tb_scanback_predictions where id_eventbrick = z.id_eventbrick and id_zone = z.id)  )" +
                    "on (id_eventbrick = idbrick and id_zone = idz)) " +
                    "inner join tb_plates on (id_eventbrick = idbrick and id_plate = id)), " +
                    "(select Z as currZ from tb_plates where id_eventbrick = " + StartupInfo.Plate.BrickId + " and id = " + StartupInfo.Plate.PlateId + ")";
            }

            HE.WriteLine("PredictionScan3Driver starting with settings:");
            HE.WriteLine(HE.ProgramSettings);

            if (StartupInfo.ExeRepository.EndsWith("\\")) StartupInfo.ExeRepository = StartupInfo.ExeRepository.Remove(StartupInfo.ExeRepository.Length - 1, 1);
            if (StartupInfo.ScratchDir.EndsWith("\\")) StartupInfo.ScratchDir = StartupInfo.ScratchDir.Remove(StartupInfo.ScratchDir.Length - 1, 1);
            if (StartupInfo.LinkedZonePath.EndsWith("\\")) StartupInfo.LinkedZonePath = StartupInfo.LinkedZonePath.Remove(StartupInfo.LinkedZonePath.Length - 1, 1);
            if (StartupInfo.RawDataPath.EndsWith("\\")) StartupInfo.RawDataPath = StartupInfo.RawDataPath.Remove(StartupInfo.RawDataPath.Length - 1, 1);

            DumpStream = new System.IO.StreamWriter(StartupInfo.ScratchDir + "\\predictionscan3driver_dump_" + StartupInfo.ProcessOperationId + "_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.Plate.PlateId + ".txt", false);
            DumpStream.AutoFlush = true;
            DumpStream.WriteLine("BRICK\tPLATE\tPATH\tTRIALS\tGRAINS\tSIGMA");

            lock(Conn)
                Trans = Conn.BeginTransaction();

            long parentopid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID_PARENT_OPERATION FROM TB_PROC_OPERATIONS WHERE ID = " + StartupInfo.ProcessOperationId, Conn, null).ExecuteScalar());

            System.Data.DataSet dps = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter(
                ProgSettings.ReadTolerancesFromPredictionTable ?
                ("SELECT ID_PATH, POSX, POSY, SLOPEX, SLOPEY, FRAME, POSTOL1, POSTOL2, SLOPETOL1, SLOPETOL2 FROM TB_SCANBACK_PREDICTIONS INNER JOIN TB_SCANBACK_PATHS ON (TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK = TB_SCANBACK_PATHS.ID_EVENTBRICK AND TB_SCANBACK_PREDICTIONS.ID_PATH = TB_SCANBACK_PATHS.ID) WHERE ID_PROCESSOPERATION = " + parentopid + " AND TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID_PLATE = " + StartupInfo.Plate.PlateId) :
                ("SELECT ID_PATH, POSX, POSY, SLOPEX, SLOPEY FROM TB_SCANBACK_PREDICTIONS INNER JOIN TB_SCANBACK_PATHS ON (TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK = TB_SCANBACK_PATHS.ID_EVENTBRICK AND TB_SCANBACK_PREDICTIONS.ID_PATH = TB_SCANBACK_PATHS.ID) WHERE ID_PROCESSOPERATION = " + parentopid + " AND TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID_PLATE = " + StartupInfo.Plate.PlateId),
                Conn, Trans).Fill(dps);

            foreach (System.Data.DataRow dxr in dps.Tables[0].Rows)
                HE.WriteLine("Prediction : " + dxr[0].ToString() + " " + SySal.OperaDb.Convert.ToDouble(dxr[1]).ToString("F1") + " " + SySal.OperaDb.Convert.ToDouble(dxr[2]).ToString("F1") + " " + SySal.OperaDb.Convert.ToDouble(dxr[3]).ToString("F4") + " " + SySal.OperaDb.Convert.ToDouble(dxr[4]).ToString("F4"));

            ProgressInfo = HE.ProgressInfo;
            ProgressInfo.StartTime = System.DateTime.Now;
            ProgressInfo.FinishTime = ProgressInfo.StartTime.AddDays(1.0);
            ProgressInfo.Progress = 0.0;
            RecalibrationDone = false;

            TotalZones = dps.Tables[0].Rows.Count;
            foreach (System.Data.DataRow dr in dps.Tables[0].Rows)
            {
                SySal.DAQSystem.Drivers.Prediction pred = new SySal.DAQSystem.Drivers.Prediction();
                pred.Series = Convert.ToInt64(dr[0]);
                pred.PredictedPosX = Convert.ToDouble(dr[1]);
                pred.PredictedPosY = Convert.ToDouble(dr[2]);
                pred.PredictedSlopeX = Convert.ToDouble(dr[3]);
                pred.PredictedSlopeY = Convert.ToDouble(dr[4]);
                PredictionsCenter.X += pred.PredictedPosX;
                PredictionsCenter.Y += pred.PredictedPosY;
                if (ProgSettings.ReadTolerancesFromPredictionTable)
                {
                    switch (Char.ToUpper(Convert.ToChar(dr[5])))
                    {
                        case 'C': pred.ToleranceFrame = SySal.DAQSystem.Frame.Cartesian; break;
                        case 'P': pred.ToleranceFrame = SySal.DAQSystem.Frame.Polar; break;
                        default: throw new Exception("Unknown frame type " + dr[5].ToString() + " for predictions.");
                    }
                    pred.PositionTolerance1 = Convert.ToDouble(dr[6]);
                    pred.PositionTolerance2 = Convert.ToDouble(dr[7]);
                    pred.SlopeTolerance1 = Convert.ToDouble(dr[6]);
                    pred.SlopeTolerance2 = Convert.ToDouble(dr[7]);
                    pred.MinX = pred.PredictedPosX - Math.Max(pred.PositionTolerance1, pred.PositionTolerance2) - 0.5 * ProgSettings.BaseThickness * Math.Max(pred.SlopeTolerance1, pred.SlopeTolerance2);
                    pred.MaxX = pred.PredictedPosX + Math.Max(pred.PositionTolerance1, pred.PositionTolerance2) - 0.5 * ProgSettings.BaseThickness * Math.Max(pred.SlopeTolerance1, pred.SlopeTolerance2);
                    pred.MinY = pred.PredictedPosY - Math.Max(pred.PositionTolerance1, pred.PositionTolerance2) - 0.5 * ProgSettings.BaseThickness * Math.Max(pred.SlopeTolerance1, pred.SlopeTolerance2);
                    pred.MaxY = pred.PredictedPosY + Math.Max(pred.PositionTolerance1, pred.PositionTolerance2) - 0.5 * ProgSettings.BaseThickness * Math.Max(pred.SlopeTolerance1, pred.SlopeTolerance2);
                }
                else
                {
                    pred.ToleranceFrame = SySal.DAQSystem.Frame.Polar;
                    pred.PositionTolerance1 = ProgSettings.PositionTolerance;
                    pred.PositionTolerance2 = ProgSettings.PositionTolerance + Math.Sqrt(pred.PredictedSlopeX * pred.PredictedSlopeX + pred.PredictedSlopeY * pred.PredictedSlopeY) * ProgSettings.PositionToleranceIncreaseWithSlope;
                    pred.SlopeTolerance1 = ProgSettings.SlopeTolerance;
                    pred.SlopeTolerance2 = ProgSettings.SlopeTolerance + Math.Sqrt(pred.PredictedSlopeX * pred.PredictedSlopeX + pred.PredictedSlopeY * pred.PredictedSlopeY) * ProgSettings.SlopeToleranceIncreaseWithSlope;
                    pred.MinX = pred.PredictedPosX - ProgSettings.PositionTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeX;
                    pred.MaxX = pred.PredictedPosX + ProgSettings.PositionTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeX;
                    pred.MinY = pred.PredictedPosY - ProgSettings.PositionTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeY;
                    pred.MaxY = pred.PredictedPosY + ProgSettings.PositionTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeY;
                }
                if (ProgSettings.EnableSlopePresetting)
                {
                    pred.UsePresetSlope = true;
                    pred.PresetSlope.X = pred.PredictedSlopeX;
                    pred.PresetSlope.Y = pred.PredictedSlopeY;
                    pred.PresetSlopeAcc.X = Math.Abs(pred.PredictedSlopeX) * ProgSettings.SlopePresetXYToleranceIncreaseWithSlope + ProgSettings.SlopePresetXYTolerance;
                    pred.PresetSlopeAcc.Y = Math.Abs(pred.PredictedSlopeY) * ProgSettings.SlopePresetXYToleranceIncreaseWithSlope + ProgSettings.SlopePresetXYTolerance;
                }
                else pred.UsePresetSlope = false;
                pred.MaxTrials = ProgSettings.MaxTrials;
                pred.Outname = StartupInfo.RawDataPath + "\\predictionscan_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.Plate.PlateId + "_" + pred.Series;
                int i;
                for (i = 0; i < ScanningResults.Count; i++)
                    if (((ScanningResult)ScanningResults[i]).PathId == pred.Series)
                        break;
                if (i == ScanningResults.Count)
                {
                    string[] rwds = System.IO.Directory.GetFiles(pred.Outname.Substring(0, pred.Outname.LastIndexOf("\\")), pred.Outname.Substring(pred.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
                    foreach (string rwd in rwds)
                        try
                        {
                            System.IO.File.Delete(rwd);
                        }
                        catch (Exception) { }
                    try
                    {
                        System.IO.File.Delete(pred.Outname + ".rwc");
                    }
                    catch (Exception) { }
                    ScanQueue.Add(pred);
                }
                else
                    WriteResult(pred, (ScanningResult)ScanningResults[i], null);
            }
            if (TotalZones > 0)
            {
                PredictionsCenter.X /= TotalZones;
                PredictionsCenter.Y /= TotalZones;
            }
            UpdatePlots();


            WorkerThread.Start();

            Exe e = new Exe();
            HE.InterruptNotifier = e;
            AppDomain.CurrentDomain.SetData(SySal.DAQSystem.Drivers.HostEnv.WebAccessString, e);        

            /* Recalibration here */

            System.Collections.ArrayList RecalQueue = new System.Collections.ArrayList();
            if (ProgSettings.RecalibrationSelectionText == null || ProgSettings.RecalibrationSelectionText.Trim().Length == 0)
            {
                RecalibrationDone = true;
                DirRecal.MXX = DirRecal.MYY = 1.0;
                DirRecal.MXY = DirRecal.MYX = 0.0;
                DirRecal.TX = DirRecal.TY = DirRecal.TZ = 0.0;
                DirRecal.RX = DirRecal.RY = 0.0;
                InvRecal.MXX = InvRecal.MYY = 1.0;
                InvRecal.MXY = InvRecal.MYX = 0.0;
                InvRecal.TX = InvRecal.TY = InvRecal.TZ = 0.0;
                InvRecal.RX = InvRecal.RY = 0.0;
                MyTransformation.Transformation = InvRecal;
                RecalibrationTracksToScan = 0;
                UpdateProgress();
            }

            if (!RecalibrationDone)
            {
                string recalsql = ProgSettings.RecalibrationSelectionText;
                recalsql = recalsql.Replace("_BRICK_", StartupInfo.Plate.BrickId.ToString()).Replace("_PLATE_", StartupInfo.Plate.PlateId.ToString());
                System.Data.DataSet rds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter(recalsql, Conn, null).Fill(rds);
                RecalibrationTracksToScan = rds.Tables[0].Rows.Count;
                if (RecalibrationTracksToScan < ProgSettings.RecalibrationMinTracks && (ProgSettings.LocalIntercalibration == IntercalibrationMode.None || RecalibrationTracksToScan != 0)) throw new Exception("Too few recalibration tracks! " + ProgSettings.RecalibrationMinTracks + " expected, " + rds.Tables[0].Rows.Count + " found.");
                if (ProgSettings.LocalIntercalibration == IntercalibrationMode.None && rds.Tables[0].Columns.Count != 4) throw new Exception("POSX POSY SLOPEX SLOPEY columns expected, " + rds.Tables[0].Columns + " found!");
                else if (ProgSettings.LocalIntercalibration != IntercalibrationMode.None && rds.Tables[0].Columns.Count != 10) throw new Exception("POSX POSY SLOPEX SLOPEY ID_ZONE ID_TRACK columns expected, " + rds.Tables[0].Columns + " found - INTERNAL ERROR!");

                System.Data.DataSet bds = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT (MINX + MAXX) * 0.5 - ZEROX AS RX, (MINY + MAXY) * 0.5 - ZEROY AS RY, MINX - ZEROX AS RMINX, MAXX - ZEROX AS RMAXX, MINY - ZEROY AS RMINY, MAXY - ZEROY AS RMAXY FROM TB_EVENTBRICKS WHERE ID = " + StartupInfo.Plate.BrickId, Conn, null).Fill(bds);
                InvRecal.RX = DirRecal.RX = Convert.ToDouble(bds.Tables[0].Rows[0][0]);
                InvRecal.RY = DirRecal.RY = Convert.ToDouble(bds.Tables[0].Rows[0][1]);
                PlateMinX = Convert.ToDouble(bds.Tables[0].Rows[0][2]);
                PlateMaxX = Convert.ToDouble(bds.Tables[0].Rows[0][3]);
                PlateMinY = Convert.ToDouble(bds.Tables[0].Rows[0][4]);
                PlateMaxY = Convert.ToDouble(bds.Tables[0].Rows[0][5]);

                int recalid = 0;
                double minx, miny, maxx, maxy, x, y;

                if (ProgSettings.LocalIntercalibration == IntercalibrationMode.None)
                {
                    minx = maxx = Convert.ToDouble(rds.Tables[0].Rows[0][0]);
                    miny = maxy = Convert.ToDouble(rds.Tables[0].Rows[0][1]);
                    foreach (System.Data.DataRow dr in rds.Tables[0].Rows)
                    {
                        x = Convert.ToDouble(dr[0]);
                        y = Convert.ToDouble(dr[1]);
                        if (minx > x) minx = x;
                        else if (maxx < x) maxx = x;
                        if (miny > y) miny = y;
                        else if (maxy < y) maxy = y;

                        recalid++;
                        SySal.DAQSystem.Drivers.Prediction pred = new SySal.DAQSystem.Drivers.Prediction();
                        pred.Series = -recalid;
                        pred.PredictedPosX = Convert.ToDouble(dr[0]);
                        pred.PredictedPosY = Convert.ToDouble(dr[1]);
                        pred.PredictedSlopeX = Convert.ToDouble(dr[2]);
                        pred.PredictedSlopeY = Convert.ToDouble(dr[3]);
                        pred.ToleranceFrame = SySal.DAQSystem.Frame.Polar;
                        pred.PositionTolerance1 = ProgSettings.RecalibrationPosTolerance;
                        pred.PositionTolerance2 = ProgSettings.RecalibrationPosTolerance + Math.Sqrt(pred.PredictedSlopeX * pred.PredictedSlopeX + pred.PredictedSlopeY * pred.PredictedSlopeY) * ProgSettings.PositionToleranceIncreaseWithSlope;
                        pred.SlopeTolerance1 = ProgSettings.SlopeTolerance;
                        pred.SlopeTolerance2 = ProgSettings.SlopeTolerance + Math.Sqrt(pred.PredictedSlopeX * pred.PredictedSlopeX + pred.PredictedSlopeY * pred.PredictedSlopeY) * ProgSettings.SlopeToleranceIncreaseWithSlope;
                        pred.MinX = pred.PredictedPosX - ProgSettings.RecalibrationPosTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeX;
                        pred.MaxX = pred.PredictedPosX + ProgSettings.RecalibrationPosTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeX;
                        pred.MinY = pred.PredictedPosY - ProgSettings.RecalibrationPosTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeY;
                        pred.MaxY = pred.PredictedPosY + ProgSettings.RecalibrationPosTolerance - 0.5 * ProgSettings.BaseThickness * pred.PredictedSlopeY;
                        pred.MaxTrials = 1;
                        pred.Outname = StartupInfo.RawDataPath + "\\predictionscan_recal_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.Plate.PlateId + "_" + recalid;
                        if (ProgSettings.EnableSlopePresetting)
                        {
                            pred.UsePresetSlope = true;
                            pred.PresetSlope.X = pred.PredictedSlopeX;
                            pred.PresetSlope.Y = pred.PredictedSlopeY;
                            pred.PresetSlopeAcc.X = Math.Abs(pred.PredictedSlopeX) * ProgSettings.SlopePresetXYToleranceIncreaseWithSlope + ProgSettings.SlopePresetXYTolerance;
                            pred.PresetSlopeAcc.Y = Math.Abs(pred.PredictedSlopeY) * ProgSettings.SlopePresetXYToleranceIncreaseWithSlope + ProgSettings.SlopePresetXYTolerance;
                        }
                        else pred.UsePresetSlope = false;

                        string[] rwds = System.IO.Directory.GetFiles(pred.Outname.Substring(0, pred.Outname.LastIndexOf("\\")), pred.Outname.Substring(pred.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
                        foreach (string rwd in rwds)
                            try
                            {
                                System.IO.File.Delete(rwd);
                            }
                            catch (Exception) { }
                        try
                        {
                            System.IO.File.Delete(pred.Outname + ".rwc");
                        }
                        catch (Exception) { }

                        RecalQueue.Add(pred);
                    }
                    if (maxx - minx < ProgSettings.RecalibrationMinXDistance) throw new Exception("Tracks found are too close in X direction!");
                    if (maxy - miny < ProgSettings.RecalibrationMinYDistance) throw new Exception("Tracks found are too close in Y direction!");
                }
                else
                {
                    if (rds.Tables[0].Rows.Count > 0)
                    {
                        RefNominalZ = SySal.OperaDb.Convert.ToDouble(rds.Tables[0].Rows[0][7]);
                        CurrNominalZ = SySal.OperaDb.Convert.ToDouble(rds.Tables[0].Rows[0][8]);
                        RefCalibZ = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand(
                            "select nvl(calib2z, nvl(calib1z, nomz)) from " +
                            "(select idb, idp, nomz, calib1z, z as calib2z from " +
                            "(select idb, idp, nomz, calib1z, id_calibration_operation from " +
                            "(select idb, idp, nomz, z as calib1z from " +
                            "(select /*+index(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, z as nomz from tb_plates where id_eventbrick = " + StartupInfo.Plate.BrickId + " and id = " + rds.Tables[0].Rows[0][9].ToString() + ") " +
                            "left join tb_plate_calibrations on (idb = id_eventbrick and idp = id_plate and id_processoperation = " + rds.Tables[0].Rows[0][6].ToString() + ")) " +
                            "left join tb_proc_operations on (id = " + rds.Tables[0].Rows[0][6].ToString() + ")) " +
                            "left join tb_plate_calibrations on (idb = id_eventbrick and idp = id_plate and id_processoperation = id_calibration_operation)) ", Conn).ExecuteScalar());
                    }
                    else
                    {
                        RefNominalZ = CurrNominalZ = RefCalibZ = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX(TB_PLATES PK_PLATES) */ Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.Plate.BrickId + " AND ID = " + StartupInfo.Plate.PlateId, Conn).ExecuteScalar());
                    }
                    IntercalTracks = new IntercalTrack[rds.Tables[0].Rows.Count];
                    SySal.BasicTypes.Rectangle IntercalRect = new SySal.BasicTypes.Rectangle();
                    if (IntercalTracks.Length > 0)
                    {
                        IntercalRect.MinX = IntercalRect.MaxX = SySal.OperaDb.Convert.ToDouble(rds.Tables[0].Rows[0][0]);
                        IntercalRect.MinY = IntercalRect.MaxY = SySal.OperaDb.Convert.ToDouble(rds.Tables[0].Rows[0][1]);
                    }
                    for (recalid = 0; recalid < rds.Tables[0].Rows.Count; recalid++)
                    {
                        System.Data.DataRow dr = rds.Tables[0].Rows[recalid];
                        IntercalTrack tk = new IntercalTrack();
                        IntercalTracks[recalid] = tk;
                        tk.IdZone = Convert.ToInt64(dr[4]);
                        tk.Id = Convert.ToInt32(dr[5]);
                        tk.Info = new SySal.Tracking.MIPEmulsionTrackInfo();
                        tk.Info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[0]);
                        tk.Info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[1]);
                        if (tk.Info.Intercept.X < IntercalRect.MinX) IntercalRect.MinX = tk.Info.Intercept.X;
                        else if (tk.Info.Intercept.X > IntercalRect.MaxX) IntercalRect.MaxX = tk.Info.Intercept.X;
                        if (tk.Info.Intercept.Y < IntercalRect.MinY) IntercalRect.MinY = tk.Info.Intercept.Y;
                        else if (tk.Info.Intercept.Y > IntercalRect.MaxY) IntercalRect.MaxY = tk.Info.Intercept.Y;
                        tk.Info.Slope.X = Convert.ToDouble(dr[2]);
                        tk.Info.Slope.Y = Convert.ToDouble(dr[3]);
                    }
                    SySal.BasicTypes.Vector2 IntercalCenter = new SySal.BasicTypes.Vector2();
                    IntercalCenter.X = 0.5 * (IntercalRect.MinX + IntercalRect.MaxX);
                    IntercalCenter.Y = 0.5 * (IntercalRect.MinY + IntercalRect.MaxY);

                    SySal.DAQSystem.Drivers.Prediction pred = new SySal.DAQSystem.Drivers.Prediction();
                    pred.Series = -1;
                    if (Math.Abs(PredictionsCenter.X - IntercalCenter.X) < 500 || IntercalTracks.Length <= 0)
                        pred.PredictedPosX = PredictionsCenter.X;
                    else
                        pred.PredictedPosX = IntercalCenter.X + 500 * Math.Sign(PredictionsCenter.X - IntercalCenter.X);
                    if (Math.Abs(PredictionsCenter.Y - IntercalCenter.Y) < 500 || IntercalTracks.Length <= 0)
                        pred.PredictedPosY = PredictionsCenter.Y;
                    else
                        pred.PredictedPosY = IntercalCenter.Y + 500 * Math.Sign(PredictionsCenter.Y - IntercalCenter.Y);
                    HE.WriteLine("IntercalCenter: " + IntercalCenter.X.ToString("F1") + "," + IntercalCenter.Y.ToString("F1"));
                    HE.WriteLine("PredictionsCenter: " + PredictionsCenter.X.ToString("F1") + "," + PredictionsCenter.Y.ToString("F1"));
                    HE.WriteLine("Scan center: " + pred.PredictedPosX.ToString("F1") + "," + pred.PredictedPosY.ToString("F1"));
                    pred.PredictedSlopeX = 0.0;
                    pred.PredictedSlopeY = 0.0;
                    pred.ToleranceFrame = SySal.DAQSystem.Frame.Cartesian;
                    pred.PositionTolerance1 = ProgSettings.RecalibrationMinXDistance;
                    pred.PositionTolerance2 = ProgSettings.RecalibrationMinYDistance;
                    pred.SlopeTolerance1 = 1.0;
                    pred.SlopeTolerance2 = 1.0;
                    pred.MinX = pred.PredictedPosX - 0.5 * ProgSettings.RecalibrationMinXDistance;
                    pred.MaxX = pred.PredictedPosX + 0.5 * ProgSettings.RecalibrationMinXDistance;
                    pred.MinY = pred.PredictedPosY - 0.5 * ProgSettings.RecalibrationMinYDistance;
                    pred.MaxY = pred.PredictedPosY + 0.5 * ProgSettings.RecalibrationMinYDistance;

                    if (pred.MinX < PlateMinX + 500.0)
                    {
                        pred.MinX = PlateMinX + 500.0;
                        pred.MaxX = pred.MinX + ProgSettings.RecalibrationMinXDistance;
                    }
                    else if (pred.MaxX > PlateMaxX - 500.0)
                    {
                        pred.MaxX = PlateMaxX - 500.0;
                        pred.MinX = pred.MaxX - ProgSettings.RecalibrationMinXDistance;
                    }
                    if (pred.MinY < PlateMinY + 500.0)
                    {
                        pred.MinY = PlateMinY + 500.0;
                        pred.MaxY = pred.MinY + ProgSettings.RecalibrationMinYDistance;
                    }
                    else if (pred.MaxY > PlateMaxY - 500.0)
                    {
                        pred.MaxY = PlateMaxY - 500.0;
                        pred.MinY = pred.MaxY - ProgSettings.RecalibrationMinYDistance;
                    }

                    pred.MaxTrials = 1;
                    pred.Outname = StartupInfo.RawDataPath + "\\predictionscan_recal_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.Plate.PlateId + "_" + recalid;
                    pred.UsePresetSlope = true;
                    pred.PresetSlope.X = pred.PresetSlope.Y = 0.0;
                    pred.PresetSlopeAcc.X = pred.PresetSlopeAcc.Y = ProgSettings.LocalIntercalibrationMaxSlope;

                    string[] rwds = System.IO.Directory.GetFiles(pred.Outname.Substring(0, pred.Outname.LastIndexOf("\\")), pred.Outname.Substring(pred.Outname.LastIndexOf("\\") + 1) + ".rwd.*");
                    foreach (string rwd in rwds)
                        try
                        {
                            System.IO.File.Delete(rwd);
                        }
                        catch (Exception) { }
                    try
                    {
                        System.IO.File.Delete(pred.Outname + ".rwc");
                    }
                    catch (Exception) { }

                    RecalQueue.Add(pred);
                }
            }

            /* End recalibration */

            HE.WriteLine("Original path length: " + PathLen(ScanQueue).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            ScanQueue = OptimizePath(ScanQueue);
            HE.WriteLine("Optimized path length: " + PathLen(ScanQueue).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            ScanQueue.InsertRange(0, RecalQueue);

            if (ScanSrv.SetScanLayout((string)new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE (ID = " + ProgSettings.ScanningConfigId + ")", Conn, Trans).ExecuteScalar()) == false)
                throw new Exception("Scan Server configuration refused!");

            if (ScanSrv.LoadPlate(StartupInfo.Plate) == false) throw new Exception("Can't load plate " + StartupInfo.Plate.PlateId + " + brick " + StartupInfo.Plate.BrickId);
            while (true)
            {
                if (ScanQueue.Count <= 0)
                {
                    ProcessEvent.WaitOne();
                    if (ThisException != null) throw ThisException;
                    if (ScanQueue.Count <= 0) break;
                }
                Prediction zd = null;
                object o = null;
                lock (ScanQueue)
                    o = ScanQueue[0];

                if (o is ManualCheck)
                {
                    ManualCheck mc = (ManualCheck)o;
                    mc.Output = ScanSrv.RequireManualCheck(mc.Input);
                    ScanQueue.RemoveAt(0);
                    WorkQueue.Enqueue(mc);
                    if (ProcessEvent.WaitOne(0, false) == true)
                    {
                        ProcessEvent.Reset();
                        WorkerThread.Interrupt();
                    }
                    continue;
                }
                else zd = (Prediction)o;
                
                if (zd.Series > 0)
                {
                    if (RecalibrationDone == false)
                    {
                        RecalEvent.WaitOne();
                        if (ThisException != null) throw ThisException;
                        if (RecalibrationDone == false) throw new Exception("Recalibration failed!");
                    }
                }
                ProgressInfo.Progress = ((double)TotalZones + RecalQueue.Count - ScanQueue.Count) / ((double)(TotalZones + RecalQueue.Count));
                HE.ProgressInfo = ProgressInfo;
                // UpdateProgress();
                string temp = ReplaceStrings(StartupInfo.RawDataPath + "\\" + zd.Outname, zd.Series, "");
                SySal.DAQSystem.Scanning.ZoneDesc wd = new SySal.DAQSystem.Scanning.ZoneDesc();
                wd.Series = zd.Series;
                if (RecalibrationDone)
                {
                    if (zd.MaxTrials == ProgSettings.MaxTrials)
                    {
                        zd.PredictedPosX += PredictionCorrectDeltaZ * zd.PredictedSlopeX;
                        zd.PredictedPosY += PredictionCorrectDeltaZ * zd.PredictedSlopeY;
                        zd.MinX += PredictionCorrectDeltaZ * zd.PredictedSlopeX;
                        zd.MaxX += PredictionCorrectDeltaZ * zd.PredictedSlopeX;
                        zd.MinY += PredictionCorrectDeltaZ * zd.PredictedSlopeY;
                        zd.MaxY += PredictionCorrectDeltaZ * zd.PredictedSlopeY;
                    }

                    double x = 0.5 * (zd.MinX + zd.MaxX);
                    double y = 0.5 * (zd.MinY + zd.MaxY);
                    double dx = zd.MaxX - zd.MinX;
                    double dy = zd.MaxY - zd.MinY;

                    wd.MinX = DirRecal.MXX * (x - DirRecal.RX) + DirRecal.MXY * (y - DirRecal.RY) + DirRecal.TX + DirRecal.RX - 0.5 * dx;
                    wd.MinY = DirRecal.MYX * (x - DirRecal.RX) + DirRecal.MYY * (y - DirRecal.RY) + DirRecal.TY + DirRecal.RY - 0.5 * dy;
                    wd.MaxX = wd.MinX + dx;
                    wd.MaxY = wd.MinY + dy;
                }
                else
                {
                    wd.MinX = zd.MinX;
                    wd.MinY = zd.MinY;
                    wd.MaxX = zd.MaxX;
                    wd.MaxY = zd.MaxY;
                }
                wd.Outname = zd.Outname;
                wd.UsePresetSlope = zd.UsePresetSlope;
                wd.PresetSlope = zd.PresetSlope;
                wd.PresetSlopeAcc = zd.PresetSlopeAcc;
                bool usepreload = false;
                SySal.BasicTypes.Rectangle nextrect = new SySal.BasicTypes.Rectangle();
                lock (ScanQueue)
                    if (ScanQueue.Count >= 2)
                    {
                        usepreload = false;
                        object no = ScanQueue[1];
                        if (no is Prediction)
                        {
                            usepreload = true;
                            SySal.DAQSystem.Drivers.Prediction nextp = (SySal.DAQSystem.Drivers.Prediction)no;
                            if (RecalibrationDone)
                            {
                                double x = 0.5 * (nextp.MinX + nextp.MaxX);
                                double y = 0.5 * (nextp.MinY + nextp.MaxY);
                                double dx = nextp.MaxX - nextp.MinX;
                                double dy = nextp.MaxY - nextp.MinY;

                                nextrect.MinX = DirRecal.MXX * (x - DirRecal.RX) + DirRecal.MXY * (y - DirRecal.RY) + DirRecal.TX + DirRecal.RX - 0.5 * dx;
                                nextrect.MinY = DirRecal.MYX * (x - DirRecal.RX) + DirRecal.MYY * (y - DirRecal.RY) + DirRecal.TY + DirRecal.RY - 0.5 * dy;
                                nextrect.MaxX = nextrect.MinX + dx;
                                nextrect.MaxY = nextrect.MinY + dy;
                            }
                            else
                            {
                                nextrect.MinX = nextp.MinX;
                                nextrect.MinY = nextp.MinY;
                                nextrect.MaxX = nextp.MaxX;
                                nextrect.MaxY = nextp.MaxY;
                            }
                        }
                    }
                if ((usepreload && !ProgSettings.DisableScanAndMoveToNext && ScanSrv.ScanAndMoveToNext(wd, nextrect)) || ScanSrv.Scan(wd))
                {
                    lock (ScanQueue)
                    {
                        ScanQueue.RemoveAt(0);
                        WorkQueue.Enqueue(zd);
                        if (ProcessEvent.WaitOne(0, false) == true)
                        {
                            ProcessEvent.Reset();
                            WorkerThread.Interrupt();
                        }
                    }
                }
                else
                {
                    if ((zd.Series < 0 && IgnoreRecalFailure) || (zd.Series >= 0 && IgnoreScanFailure))
                    {
                        zd.CandidateIndex = ScanFailed;
                        zd.CandidateInfo = null;
                        lock (ScanQueue)
                        {
                            ScanQueue.RemoveAt(0);
                            WorkQueue.Enqueue(zd);
                            if (ProcessEvent.WaitOne(0, false) == true)
                            {
                                ProcessEvent.Reset();
                                WorkerThread.Interrupt();
                            }
                        }

                    }
                    else throw new Exception("Scanning failed for zone " + zd.Series + " plate " + StartupInfo.Plate.PlateId + " brick " + StartupInfo.Plate.BrickId);
                }

                System.TimeSpan timeelapsed = System.DateTime.Now - ProgressInfo.StartTime;
                ProgressInfo.FinishTime = ProgressInfo.StartTime.AddMilliseconds(timeelapsed.TotalMilliseconds / ((TotalZones + RecalQueue.Count - ScanQueue.Count) + 1) * (TotalZones + RecalQueue.Count));
            }

            lock (WorkQueue)
            {
                WorkQueue.Clear();
                WorkerThread.Interrupt();
            }
            WorkerThread.Join();

            HE.Progress = 1.0;
            HE.InterruptNotifier = null;

            DumpStream.Close();

            lock(Conn)
                
            lock (Conn)
            {
                try
                {
                    Trans.Commit();
                    if (ProgSettings.ResultFile != null && ProgSettings.ResultFile.Length > 0)
                    {
                        string resfname = StartupInfo.ScratchDir;
                        if (resfname.EndsWith("\\") == false && resfname.EndsWith("/") == false) resfname += "\\";
                        resfname += ProgSettings.ResultFile + "_" + StartupInfo.Plate.BrickId + "_" + 
                            new SySal.OperaDb.OperaDbCommand("SELECT NVL(ID_PARENT_OPERATION, ID) AS IDOP FROM TB_PROC_OPERATIONS WHERE ID = " + StartupInfo.ProcessOperationId, Conn).ExecuteScalar().ToString() +
                            ".txt";
                        System.IO.File.AppendAllText(resfname, ResultText);
                    }
                }
                finally
                {
                    Conn.Close();
                    Conn = null;
                }
            }
            ProgressInfo.Complete = true;
            ProgressInfo.ExitException = null;
            ProgressInfo.Progress = 1.0;
            ProgressInfo.FinishTime = System.DateTime.Now;
            HE.ProgressInfo = ProgressInfo;
            UpdatePlots();
        }


        private static string ReplaceStrings(string s, long zoneid, string name)
        {
            string ns = (string)s.Clone();
            ns = ns.Replace("%EXEREP%", StartupInfo.ExeRepository);
            ns = ns.Replace("%RWDDIR%", StartupInfo.RawDataPath);
            ns = ns.Replace("%TLGDIR%", StartupInfo.LinkedZonePath);
            ns = ns.Replace("%SCRATCH%", StartupInfo.ScratchDir);
            ns = ns.Replace("%RWD%", StartupInfo.RawDataPath + name);
            ns = ns.Replace("%TLG%", StartupInfo.LinkedZonePath + name);
            ns = ns.Replace("%ZONEID%", zoneid.ToString("X8"));
            return ns;
        }

        internal static string SyntaxHelp()
        {
            return "Known parameters for selection function: \r\n" +
                "N     -> Grains \r\n" +
                "A     -> AreaSum \r\n" +
                "S     -> Sigma \r\n" +
                "PPX/Y -> Predicted X/Y position\r\n" +
                "PSX/Y -> Predicted X/Y slope\r\n" +
                "PSL   -> Predicted slope\r\n" +
                "FPX/Y -> Found X/Y position\r\n" +
                "FSX/Y -> Found X/Y slope\r\n" +
                "FSL   -> Found slope\r\n" +
                "TX/Y  -> X/Y component of transverse unit vector\r\n" +
                "LX/Y  -> X/Y component of longitudinal unit vector\r\n" +
                "DPX/Y -> Found - Predicted X/Y position\r\n" +
                "DSX/Y -> Found - Predicted X/Y slope\r\n" +
                "DPT/L -> Found - Predicted Transverse/Longitudinal position\r\n" +
                "DST/L -> Found - Predicted Transverse/Longitudinal slope\r\n" +
                "TRIALS -> Number of remaining trials for the current prediction";
        }

        internal static NumericalTools.CStyleParsedFunction MakeFunctionAndCheck(string text)
        {
            NumericalTools.CStyleParsedFunction f = new NumericalTools.CStyleParsedFunction(text);
            string[] parameters = f.ParameterList;
            foreach (string s in parameters)
            {
                if (String.Compare(s, "N", true) == 0) continue;
                if (String.Compare(s, "A", true) == 0) continue;
                if (String.Compare(s, "S", true) == 0) continue;
                if (String.Compare(s, "PPX", true) == 0) continue;
                if (String.Compare(s, "PPY", true) == 0) continue;
                if (String.Compare(s, "PSX", true) == 0) continue;
                if (String.Compare(s, "PSY", true) == 0) continue;
                if (String.Compare(s, "PSL", true) == 0) continue;
                if (String.Compare(s, "FPX", true) == 0) continue;
                if (String.Compare(s, "FPY", true) == 0) continue;
                if (String.Compare(s, "FSX", true) == 0) continue;
                if (String.Compare(s, "FSY", true) == 0) continue;
                if (String.Compare(s, "FSL", true) == 0) continue;
                if (String.Compare(s, "TX", true) == 0) continue;
                if (String.Compare(s, "TY", true) == 0) continue;
                if (String.Compare(s, "LX", true) == 0) continue;
                if (String.Compare(s, "LY", true) == 0) continue;
                if (String.Compare(s, "DPX", true) == 0) continue;
                if (String.Compare(s, "DPY", true) == 0) continue;
                if (String.Compare(s, "DPT", true) == 0) continue;
                if (String.Compare(s, "DPL", true) == 0) continue;
                if (String.Compare(s, "DSX", true) == 0) continue;
                if (String.Compare(s, "DSY", true) == 0) continue;
                if (String.Compare(s, "DST", true) == 0) continue;
                if (String.Compare(s, "DSL", true) == 0) continue;
                if (String.Compare(s, "TRIALS", true) == 0) continue;
                throw new Exception("Parameter \"" + s + "\" is unknown.");
            }
            return f;
        }

        private static System.DateTime NextPlotTime;

        private static void UpdatePlots()
        {
            lock (StartupInfo)
            {
                int i, j;
                int total = 0, searched = 0, found = 0;
                double percent = 0.0;
                total = TotalZones;
                searched = ScanningResults.Count;
                if (searched < total && IsFirstPlot)
                {
                    NextPlotTime = System.DateTime.Now.AddSeconds(10.0);
                    IsFirstPlot = false;
                    return;
                }
                System.DateTime now = System.DateTime.Now;
                if (searched < total && now < NextPlotTime) return;
                NextPlotTime = now.AddSeconds(10.0);
                for (i = 0; i < ScanningResults.Count; i++)
                {
                    ScanningResult res = (ScanningResult)ScanningResults[i];
                    if (res.CandidateId > 0) found++;
                }
                double[] ppx = new double[found];
                double[] ppy = new double[found];
                double[] psx = new double[found];
                double[] psy = new double[found];
                double[] dpx = new double[found];
                double[] dpy = new double[found];
                double[] dsx = new double[found];
                double[] dsy = new double[found];
                for (i = j = 0; j < ScanningResults.Count; j++)
                {
                    ScanningResult res = (ScanningResult)ScanningResults[j];
                    if (res.CandidateId > 0)
                    {
                        ppx[i] = res.PPX;
                        ppy[i] = res.PPY;
                        psx[i] = res.PSX;
                        psy[i] = res.PSY;
                        dpx[i] = res.DPX;
                        dpy[i] = res.DPY;
                        dsx[i] = res.DSX;
                        dsy[i] = res.DSY;
                        i++;
                    }
                }
                percent = (double)found / (double)searched * 100.0;


                try
                {
                    if (found > 1)
                    {
                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = dpx;
                        gPlot.SetXDefaultLimits = false;
                        gPlot.DX = (float)(0.5f * ProgSettings.PositionTolerance / Math.Sqrt(found));
                        if (gPlot.DX < 1.0f) gPlot.DX = 1.0f;
                        gPlot.MinX = (float)(-2.0 * ProgSettings.PositionTolerance);
                        gPlot.MaxX = (float)(2.0 * ProgSettings.PositionTolerance);
                        gPlot.XTitle = "DPX (micron)";
                        gPlot.PanelFormat = "F2";
                        try
                        {
                            gPlot.HistoFit = -2;
                            gPlot.Histo(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_00.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = ppx;
                        gPlot.XTitle = "PPX (micron)";
                        gPlot.VecY = dpx;
                        gPlot.SetXDefaultLimits = true;
                        gPlot.SetYDefaultLimits = false;
                        gPlot.DY = 1.0f;
                        gPlot.YTitle = "DPX (micron)";
                        gPlot.MinY = (float)(-2.0 * ProgSettings.PositionTolerance);
                        gPlot.MaxY = (float)(2.0 * ProgSettings.PositionTolerance);
                        gPlot.PanelFormat = "F6";
                        try
                        {
                            gPlot.ScatterFit = 0;
                            gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_01.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = ppy;
                        gPlot.XTitle = "PPY (micron)";
                        gPlot.SetXDefaultLimits = true;
                        gPlot.SetYDefaultLimits = false;
                        gPlot.MinY = (float)(-2.0 * ProgSettings.PositionTolerance);
                        gPlot.MaxY = (float)(2.0 * ProgSettings.PositionTolerance);
                        gPlot.PanelFormat = "F6";
                        try
                        {
                            gPlot.ScatterFit = 0;
                            gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_02.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = dsx;
                        gPlot.SetXDefaultLimits = false;
                        gPlot.DX = (float)(0.5f * ProgSettings.SlopeTolerance / Math.Sqrt(found));
                        if (gPlot.DX < 0.001f) gPlot.DX = 0.001f;
                        gPlot.MinX = (float)(-2.0 * ProgSettings.SlopeTolerance);
                        gPlot.MaxX = (float)(2.0 * ProgSettings.SlopeTolerance);
                        gPlot.XTitle = "DSX (micron)";
                        gPlot.PanelFormat = "F5";
                        try
                        {
                            gPlot.HistoFit = -2;
                            gPlot.Histo(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_03.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = dpy;
                        gPlot.SetXDefaultLimits = false;
                        gPlot.DX = (float)(0.5f * ProgSettings.PositionTolerance / Math.Sqrt(found));
                        if (gPlot.DX < 1.0f) gPlot.DX = 1.0f;
                        gPlot.MinX = (float)(-2.0 * ProgSettings.PositionTolerance);
                        gPlot.MaxX = (float)(2.0 * ProgSettings.PositionTolerance);
                        gPlot.XTitle = "DPY (micron)";
                        gPlot.PanelFormat = "F2";
                        try
                        {
                            gPlot.HistoFit = -2;
                            gPlot.Histo(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_04.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = ppx;
                        gPlot.XTitle = "PPX (micron)";
                        gPlot.VecY = dpy;
                        gPlot.SetXDefaultLimits = true;
                        gPlot.SetYDefaultLimits = false;
                        gPlot.DY = 1.0f;
                        gPlot.MinY = (float)(-2.0 * ProgSettings.PositionTolerance);
                        gPlot.MaxY = (float)(2.0 * ProgSettings.PositionTolerance);
                        gPlot.YTitle = "DPY (micron)";
                        gPlot.PanelFormat = "F6";
                        try
                        {
                            gPlot.ScatterFit = 0;
                            gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_05.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = ppy;
                        gPlot.XTitle = "PPY (micron)";
                        gPlot.SetXDefaultLimits = true;
                        gPlot.SetYDefaultLimits = false;
                        gPlot.MinY = (float)(-2.0 * ProgSettings.PositionTolerance);
                        gPlot.MaxY = (float)(2.0 * ProgSettings.PositionTolerance);
                        gPlot.PanelFormat = "F6";
                        try
                        {
                            gPlot.ScatterFit = 0;
                            gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_06.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }

                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = dsy;
                        gPlot.SetXDefaultLimits = false;
                        gPlot.DX = (float)(0.5f * ProgSettings.SlopeTolerance / Math.Sqrt(found));
                        if (gPlot.DX < 0.001f) gPlot.DX = 0.001f;
                        gPlot.MinX = (float)(-2.0 * ProgSettings.SlopeTolerance);
                        gPlot.MaxX = (float)(2.0 * ProgSettings.SlopeTolerance);
                        gPlot.XTitle = "DSY (micron)";
                        gPlot.PanelFormat = "F5";
                        try
                        {
                            gPlot.HistoFit = -2;
                            gPlot.Histo(gMon, gIm.Width, gIm.Height);
                            gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_07.png", System.Drawing.Imaging.ImageFormat.Png);
                        }
                        catch (Exception) { }
                    }
                }
                catch (Exception) { }

                System.IO.StreamWriter w = null;
                try
                {
                    w = new System.IO.StreamWriter(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_progress.htm");
                    w.WriteLine(
                        "<html><head>" + ((searched < total) ? "<meta http-equiv=\"REFRESH\" content=\"30\">" : "") + "<title>PredictionScan3Driver Monitor</title></head><body>\r\n" +
                        "<div align=center><p><font face=\"Arial, Helvetica\" size=4 color=4444ff>PredictionScan3Driver Brick #" + StartupInfo.Plate.BrickId + ", Plate #" + StartupInfo.Plate.PlateId + "<br>Operation ID = " + StartupInfo.ProcessOperationId + "</font><hr></p></div>\r\n" +
                        "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Total = " + total + "<br>Searched = " + searched + "<br>Found = " + found + " (" + percent.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +
                        "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=000066>IgnoreScanFailure = " + IgnoreScanFailure + "<br>IgnoreRecalFailure = " + IgnoreRecalFailure + "</font></p>\r\n" +
                        "<table border=1 align=center>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_00.png\" border=0></td><td><img src=\"" + StartupInfo.ProcessOperationId + "_04.png\" border=0></td></tr>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_01.png\" border=0></td><td><img src=\"" + StartupInfo.ProcessOperationId + "_05.png\" border=0></td></tr>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_02.png\" border=0></td><td><img src=\"" + StartupInfo.ProcessOperationId + "_06.png\" border=0></td></tr>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_03.png\" border=0></td><td><img src=\"" + StartupInfo.ProcessOperationId + "_07.png\" border=0></td></tr>\r\n" +
                        "</table></div></body></html>"
                        );
                    w.Flush();
                    w.Close();
                }
                catch (Exception)
                {
                    if (w != null) w.Close();
                }
                IsFirstPlot = false;
            }
        }

        private static void UpdateProgress()
        {
            string xmlstr = "\r\n\t\t[InfoContainer]\r\n\t\t\t[RecalibrationDone]" + RecalibrationDone + "[/RecalibrationDone]" +
                "\r\n\t\t\t[DMXX]" + DirRecal.MXX + "[/DMXX]\r\n\t\t\t[DMXY]" + DirRecal.MXY + "[/DMXY]\r\n\t\t\t[DMYX]" + DirRecal.MYX + "[/DMYX]\r\n\t\t\t[DMYY]" + DirRecal.MYY + "[/DMYY]\r\n\t\t\t[DTX]" + DirRecal.TX + "[/DTX]\r\n\t\t\t[DTY]" + DirRecal.TY + "[/DTY]\r\n\t\t\t[DRX]" + DirRecal.RX + "[/DRX]\r\n\t\t\t[DRY]" + DirRecal.RY + "[/DRY]" +
                "\r\n\t\t\t[IMXX]" + InvRecal.MXX + "[/IMXX]\r\n\t\t\t[IMXY]" + InvRecal.MXY + "[/IMXY]\r\n\t\t\t[IMYX]" + InvRecal.MYX + "[/IMYX]\r\n\t\t\t[IMYY]" + InvRecal.MYY + "[/IMYY]\r\n\t\t\t[ITX]" + InvRecal.TX + "[/ITX]\r\n\t\t\t[ITY]" + InvRecal.TY + "[/ITY]\r\n\t\t\t[IRX]" + InvRecal.RX + "[/IRX]\r\n\t\t\t[IRY]" + InvRecal.RY + "[/IRY]" +
                "\r\n\t\t\t[Results]";
            xmlstr += "[/Results]\r\n\t\t[/InfoContainer]\r\n";
            ProgressInfo.CustomInfo = xmlstr;
            HE.ProgressInfo = ProgressInfo;
        }

        private static void ComputeInverse()
        {
            double idet = 1.0 / (DirRecal.MXX * DirRecal.MYY - DirRecal.MXY * DirRecal.MYX);

            InvRecal.MXX = idet * DirRecal.MYY;
            InvRecal.MXY = -idet * DirRecal.MXY;
            InvRecal.MYX = -idet * DirRecal.MYX;
            InvRecal.MYY = idet * DirRecal.MXX;
            InvRecal.TX = (DirRecal.MXY * DirRecal.TY - DirRecal.MYY * DirRecal.TX) * idet;
            InvRecal.TY = (DirRecal.MYX * DirRecal.TX - DirRecal.MXX * DirRecal.TY) * idet;
        }

        private static void ComputeRecalibration()
        {
            int valid = 0;
            double minx = 0.0, maxx = 0.0, miny = 0.0, maxy = 0.0;
            foreach (ScanningResult res in RecalibrationResults)
                if (res.CandidateId > 0)
                {
                    if (valid == 0)
                    {
                        minx = maxx = res.PPX;
                        miny = maxy = res.PPY;
                    }
                    else
                    {
                        if (minx > res.PPX) minx = res.PPX;
                        else if (maxx < res.PPX) maxx = res.PPX;
                        if (miny > res.PPY) miny = res.PPY;
                        else if (maxy < res.PPY) maxy = res.PPY;
                    }
                    valid++;
                }
            if (valid < ProgSettings.RecalibrationMinTracks) throw new Exception("Too few recalibration tracks! " + ProgSettings.RecalibrationMinTracks + " expected, " + valid + " found.");
            if (maxx - minx < ProgSettings.RecalibrationMinXDistance) throw new Exception("Tracks found are too close in X direction!");
            if (maxy - miny < ProgSettings.RecalibrationMinYDistance) throw new Exception("Tracks found are too close in Y direction!");
            double[] PX = new double[valid];
            double[] PY = new double[valid];
            double[] DX = new double[valid];
            double[] DY = new double[valid];
            valid = 0;
            foreach (ScanningResult res in RecalibrationResults)
                if (res.CandidateId > 0)
                {
                    PX[valid] = res.PPX - DirRecal.RX;
                    DX[valid] = res.DPX;
                    PY[valid] = res.PPY - DirRecal.RY;
                    DY[valid] = res.DPY;
                    valid++;
                }
            double[] outpar = new double[7];
            NumericalTools.Fitting.Affine(DX, DY, PX, PY, ref outpar);

            DirRecal.MXX = 1.0 + outpar[0];
            DirRecal.MXY = outpar[1];
            DirRecal.MYX = outpar[2];
            DirRecal.MYY = 1.0 + outpar[3];
            DirRecal.TX = outpar[4];
            DirRecal.TY = outpar[5];

            ComputeInverse();

            MyTransformation.Transformation = InvRecal;

            RecalibrationDone = true;
            UpdateProgress();
        }

        private static System.Text.RegularExpressions.Regex TwoEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s*");


        #region IInterruptNotifier Members

        /// <summary>
        /// Notifies incoming interrupts.
        /// </summary>
        /// <param name="nextint">the next interrupt to be processed.</param>
        public void NotifyInterrupt(Interrupt nextint)
        {
            lock (StartupInfo)
            {
                HE.WriteLine("Processing interrupt string:\n" + nextint.Data);
                if (nextint.Data != null && nextint.Data.Length > 0)
                {
                    string[] lines = nextint.Data.Split(',');
                    foreach (string line in lines)
                    {
                        System.Text.RegularExpressions.Match m = TwoEx.Match(line);
                        if (m.Success)
                        {
                            if (String.Compare(m.Groups[1].Value, "IgnoreScanFailure", true) == 0)
                            {
                                try
                                {
                                    IgnoreScanFailure = Convert.ToBoolean(m.Groups[2].Value);
                                    HE.WriteLine("IgnoreScanFailure = " + IgnoreScanFailure);
                                }
                                catch (Exception) { }
                            }
                            else if (String.Compare(m.Groups[1].Value, "IgnoreRecalFailure", true) == 0)
                            {
                                try
                                {
                                    IgnoreRecalFailure = Convert.ToBoolean(m.Groups[2].Value);
                                    HE.WriteLine("IgnoreRecalFailure = " + IgnoreRecalFailure);
                                }
                                catch (Exception) { }
                            }
                        }
                    }
                }
                HE.LastProcessedInterruptId = nextint.Id;
            }
        }
        #endregion

        #region IWebApplication Members

        public string ApplicationName
        {
            get { return "PredictionScan3Driver"; }
        }

        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        const string IgnoreScanFTrue = "ist";
        const string IgnoreScanFFalse = "isf";
        const string IgnoreRecalFTrue = "irt";
        const string IgnoreRecalFFalse = "irf";

        /// <summary>
        /// Processes HTTP POST method calls.
        /// </summary>
        /// <param name="sess">the user session.</param>
        /// <param name="page">the page requested (ignored).</param>
        /// <param name="postfields">commands sent to the page.</param>
        /// <returns>an HTML string with the page to be shown.</returns>
        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
        {
            string xctext = null;
            if (postfields != null)
            {
                Interrupt i = new Interrupt();
                i.Id = 0;
                foreach (string s in postfields)
                {
                    if (s.StartsWith(IgnoreRecalFFalse))
                    {
                        i.Data = "IgnoreRecalFailure False";                        
                    }
                    else if (s.StartsWith(IgnoreRecalFTrue))
                    {
                        i.Data = "IgnoreRecalFailure True";
                    }
                    if (s.StartsWith(IgnoreScanFFalse))
                    {
                        i.Data = "IgnoreScanFailure False";
                    }
                    else if (s.StartsWith(IgnoreScanFTrue))
                    {
                        i.Data = "IgnoreScanFailure True";
                    }
                }
                if (i.Data != null)
                    try
                    {
                        NotifyInterrupt(i);
                    }
                    catch (Exception x)
                    {
                        xctext = x.ToString();
                    }
            }
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                "<head>\r\n" +
                "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                "    <title>PredictionScan3Driver - " + StartupInfo.Plate.BrickId + "/" + StartupInfo.Plate.PlateId + "/" + StartupInfo.ProcessOperationId + "</title>\r\n" +
                "    <style type=\"text/css\">\r\n" +
                "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                "    </style>\r\n" +
                "</head>\r\n" +
                "<body>\r\n" +
                " <div>PredictionScan3Driver = " + StartupInfo.ProcessOperationId + "<br>Brick = " + StartupInfo.Plate.BrickId + "<br>Plate = " + StartupInfo.Plate.PlateId + "<br>IgnoreScanFailure = " + IgnoreScanFailure + "<br>IgnoreRecalFailure = " + IgnoreRecalFailure + "</div>\r\n<hr>\r\n" +
                ((xctext != null) ? "<div>Interrupt Error:<br><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "<font></div>\r\n" : "") +
                " <form action=\"" + page + "\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n" +
                "  <div>\r\n" +
                "   <input id=\"" + IgnoreRecalFFalse + "\" name=\"" + IgnoreRecalFFalse + "\" type=\"submit\" value=\"Stop on Recal failure\"/>&nbsp;<input id=\"" + IgnoreRecalFTrue + "\" name=\"" + IgnoreRecalFTrue + "\" type=\"submit\" value=\"Ignore Recal failure\"/><br>\r\n" +
                "   <input id=\"" + IgnoreScanFFalse + "\" name=\"" + IgnoreScanFFalse + "\" type=\"submit\" value=\"Stop on Scan failure\"/>&nbsp;<input id=\"" + IgnoreScanFTrue + "\" name=\"" + IgnoreScanFTrue + "\" type=\"submit\" value=\"Ignore Scan failure\"/><br>\r\n" +
                "  </div>\r\n" +
                " </form>\r\n" +
                "</body>\r\n" +
                "</html>";

            return new SySal.Web.HTMLResponse(html);
        }

        public bool ShowExceptions
        {
            get { return true; }
        }

        #endregion
    }
}
