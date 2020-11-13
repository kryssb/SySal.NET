using System;
using SySal;
using SySal.BasicTypes;

namespace SySal.DAQSystem
{
	/// <summary>
	/// ScanServerRemoteClass: performs remote scanning according to requests.
	/// </summary>
	public class ScanServer : MarshalByRefObject
	{
		/// <summary>
		/// Builds a new ScanServer.
		/// </summary>
		public ScanServer()
		{
			//
			// TODO: Add constructor logic here
			//
			throw new System.Exception("This is a stub only. Use remote activation.");
		}

		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>null to obtain an everlasting ScanServer.</returns>
		public override object InitializeLifetimeService()
		{
			return null;	
		}

		/// <summary>
		/// Starts scanning a zone.
		/// </summary>
		/// <param name="zone">zone description.</param>
		/// <returns>true if the zone was successfully scanned, false otherwise.</returns>
		public bool Scan(SySal.DAQSystem.Scanning.ZoneDesc zone)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");		
		}

		/// <summary>
		/// Starts scanning a zone, preparing to move to the next zone at the end.
		/// </summary>
		/// <param name="zone">zone description.</param>
		/// <param name="nextzone">zone to be scanned after this.</param>
		/// <returns>true if the zone was successfully scanned, false otherwise.</returns>
		public bool ScanAndMoveToNext(SySal.DAQSystem.Scanning.ZoneDesc zone, SySal.BasicTypes.Rectangle nextzone)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");		
		}

		/// <summary>
		/// Requests the ScanServer to load a plate onto a microscope stage.
		/// If a plate is already on the stage, it is unloaded if it is not the desired one.
		/// </summary>
		/// <param name="plate">plate description.</param>
		/// <returns>true if the plate was successfully loaded, false otherwise.</returns>
		public bool LoadPlate(SySal.DAQSystem.Scanning.MountPlateDesc plate)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");		
		}

		/// <summary>
		/// Requests the ScanServer to unload a plate from a microscope stage.
		/// </summary>
		/// <returns>true if the plate was successfully unloaded or the stage was empty, false otherwise.</returns>
		public bool UnloadPlate()
		{
			throw new System.Exception("This is a stub only. Use remote activation.");		
		}

		/// <summary>
		/// Tests the communication with the ScanServer.
		/// </summary>
		/// <param name="h">communication parameter.</param>
		/// <returns>true if h is 0, false otherwise.</returns>
		public bool TestComm(int h)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");
		}

		/// <summary>
		/// Alters the configuration of a specified object by changing a single parameter.
		/// </summary>
		/// <param name="objectname">name of the object whose configuration has to be changed.</param>
		/// <param name="parametername">name of the parameter to be changed.</param>
		/// <param name="parametervalue">new value to be assigned to the selected parameter.</param>
		/// <returns>true if the parameter was successfully changed, false otherwise.</returns>
		public bool SetSingleParameter(string objectname, string parametername, string parametervalue)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");
		}

		/// <summary>
		/// Sets the configuration of a specified object.
		/// </summary>
		/// <param name="objectname">name of the object whose configuration has to be changed.</param>
		/// <param name="xmlconfig">XML configuration element containing the configuration to be applied.</param>
		/// <returns>true if the configuration was successfully set, false otherwise.</returns>
		public bool SetObjectConfiguration(string objectname, string xmlconfig)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");
		}

		/// <summary>
		/// Sets the configuration of a Scan Server
		/// </summary>
		/// <param name="xmllayout">XML layout element containing the layout, connection and configurations to be used for scanning.</param>
		/// <returns>true if the layout was successfully set up, false otherwise.</returns>
		public bool SetScanLayout(string xmllayout)
		{
			throw new System.Exception("This is a stub only. Use remote activation.");
		}

		/// <summary>
		/// Tells whether the Scan Server is busy scanning any area.
		/// </summary>
		public bool IsBusy
		{
			get { throw new System.Exception("This is a stub only. Use remote activation."); }
		}

		/// <summary>
		/// Tells whether the Scan Server has a plate loaded.
		/// </summary>
		public bool IsLoaded
		{
			get { throw new System.Exception("This is a stub only. Use remote activation."); }
		}

		/// <summary>
		/// The zone currently being scanned. An exception is thrown if no zone is being scanned.
		/// </summary>
		public long CurrentZone
		{
			get { throw new System.Exception("This is a stub only. Use remote activation."); }
		}

		/// <summary>
		/// The plate currently loaded. An exception is thrown if no plate is loaded.
		/// </summary>
		public SySal.DAQSystem.Scanning.MountPlateDesc CurrentPlate
		{
			get { throw new System.Exception("This is a stub only. Use remote activation."); }
		}

        /// <summary>
        /// Requires a human operator to perform a manual check on a base track.
        /// </summary>
        /// <param name="inputbasetrack">the information about the base track to be searched.</param>
        /// <returns>the result of the manual check.</returns>
        public Scanning.ManualCheck.OutputBaseTrack RequireManualCheck(Scanning.ManualCheck.InputBaseTrack inputbasetrack)
        {
            throw new Exception("This is a stub only. Use remote activation.");
        }

        /// <summary>
        /// Measures fog and top/bottom/base thickness of a plate.
        /// </summary>
        /// <returns>the measured fog and thickness set.</returns>
        public Scanning.PlateQuality.FogThicknessSet GetFogAndThickness()
        {
            throw new Exception("This is a stub only. Use remote activation.");
        }

        /// <summary>
        /// Performs an image dump in a specified position, marking the identifier and slope of the track possibly contained.
        /// </summary>
        /// <param name="imdumpreq">the information to perform the image dump.</param>
        /// <returns><c>true</c> if the image sequence has been dumped, <c>false</c> otherwise.</returns>
        public bool ImageDump(Scanning.ImageDumpRequest imdumpreq)
        {
            throw new Exception("This is a stub only. Use remote activation.");
        }        
	}
}
