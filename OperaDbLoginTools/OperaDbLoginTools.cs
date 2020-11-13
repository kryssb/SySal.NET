using System;
using System.Security;
using SySal.OperaDb;

namespace SySal.OperaDb
{
	/// <summary>
	/// Credentials are username/password pairs.
	/// </summary>
	public sealed class OperaDbCredentials
	{
        private static System.Security.Cryptography.Rijndael Rijn = rijninit();

        private static System.Security.Cryptography.Rijndael rijninit()
        {
            System.Security.Cryptography.Rijndael rijn = System.Security.Cryptography.Rijndael.Create();
            byte[] ivblock = new byte[rijn.IV.Length];
            string computername = System.Environment.MachineName;
            int i;
            for (i = 0; i < ivblock.Length; i++)
                ivblock[i] = System.Convert.ToByte(computername[i % computername.Length]);
            rijn.IV = ivblock;
            byte[] keyblock = new byte[rijn.Key.Length];
            string path = System.Environment.UserName;
            for (i = 0; i < keyblock.Length; i++)
                keyblock[i] = System.Convert.ToByte(path[i % path.Length]);
            rijn.Key = keyblock;
            return rijn;
        }

        /// <summary>
        /// Decodes an encrypted string.
        /// </summary>
        /// <param name="instr">string to be decoded.</param>
        /// <returns>the decoded string.</returns>
        public static string Decode(string instr)
        {
            int i;
            byte[] in_array = new byte[instr.Length / 2];
            for (i = 0; i < in_array.Length; i++)
                in_array[i] = System.Convert.ToByte(((System.Convert.ToByte(instr[i * 2]) - 65) << 4) + (System.Convert.ToByte(instr[i * 2 + 1]) - 65));
            byte[] out_array = Rijn.CreateDecryptor().TransformFinalBlock(in_array, 0, in_array.Length);
            string outstr = "";
            for (i = 0; i < out_array.Length; i++) outstr += System.Convert.ToChar(out_array[i]);
            return outstr;
        }

        /// <summary>
        /// Encodes an encrypted string.
        /// </summary>
        /// <param name="instr">the string to be encrypted.</param>
        /// <returns>the encoded string.</returns>
        public static string Encode(string instr)
        {
            int i;
            byte[] in_array = new byte[instr.Length];
            for (i = 0; i < instr.Length; i++)
                in_array[i] = System.Convert.ToByte(instr[i]);
            byte[] out_array = Rijn.CreateEncryptor().TransformFinalBlock(in_array, 0, in_array.Length);
            string outstr = "";
            for (i = 0; i < out_array.Length; i++)
            {
                outstr += System.Convert.ToChar(((out_array[i] & 0xf0) >> 4) + 65);
                outstr += System.Convert.ToChar((out_array[i] & 0x0f) + 65);
            }
            return outstr;            
        }

		/// <summary>
		/// The path of the credential file.
		/// </summary>
		private static string Path { get { return System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData) + "\\sysopdbcred.dat"; } }
		/// <summary>
		/// DB server to log on to.
		/// </summary>
		public string DBServer;
		/// <summary>
		/// Opera DB user name.
		/// </summary>
        public string DBUserName;
		/// <summary>
		/// Opera DB password.
		/// </summary>
        public string DBPassword;
		/// <summary>
		/// Opera user name.
		/// </summary>
        public string OPERAUserName;
		/// <summary>
		/// Opera password.
		/// </summary>
        public string OPERAPassword;
		/// <summary>
		/// Records credentials for later use.
		/// </summary>
		public void Record()
		{
            if (System.IO.File.Exists(Path))
                System.IO.File.SetAttributes(Path, System.IO.FileAttributes.Normal);
            System.IO.File.WriteAllText(Path, Encode(DBServer) + "\n" + Encode(DBUserName) + "\n" + Encode(DBPassword) + "\n" + Encode(OPERAUserName) + "\n" + Encode(OPERAPassword));
			System.IO.File.SetAttributes(Path, System.IO.FileAttributes.Hidden);
		}
        /// <summary>
        /// Records a set of credentials to an environment variable.
        /// </summary>
        /// <param name="dict">the dictionary where this information is to be recorded.</param>
        public void RecordToEnvironment(System.Collections.Specialized.StringDictionary dict)
        {
            dict.Add(SySalEnvironmentKey, Encode(DBServer )+ " " + Encode(DBUserName) + " " + Encode(DBPassword) + " " + Encode(OPERAUserName) + " " + Encode(OPERAPassword));
        }
        /// <summary>
        /// Removes a set of credentials from an environment variable.
        /// </summary>
        public void RemoveFromEnvironment(System.Collections.Specialized.StringDictionary dict)
        {
            dict.Remove(SySalEnvironmentKey);
        }
        /// <summary>
		/// Checks database access with these credentials.
		/// </summary>
		/// <returns>the user id associated to these credentials if successful; otherwise, an exception is thrown.</returns>
		public long CheckDbAccess()
		{
			SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(DBServer, DBUserName, DBPassword);
			conn.Open();
			long uid = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(OPERAUserName, OPERAPassword, conn, null);
			conn.Close();			
			if (uid <= 0) throw new Exception("DB access credentials OK\r\nOPERA credentials refused (wrong username/password pair)");
			return uid;
		}
		/// <summary>
		/// Flushes the credentials from the record.
		/// </summary>
		private void Flush()
		{
			try
			{
				System.IO.File.Delete(Path);
			}
			catch (Exception) {}
		}
		/// <summary>
		/// Creates a new, empty set of credentials.
		/// </summary>
		public OperaDbCredentials()
		{
			DBServer = "";
			DBUserName = "";
			DBPassword = "";
			OPERAUserName = "";
			OPERAPassword = "";
		}

		/// <summary>
		/// Creates a new OperaDbConnection using the DB login information in the set of credentials.
		/// </summary>
		/// <returns>a new OperaDbConnection (not open yet) to the Opera DB specified in the credentials.</returns>
		public OperaDbConnection Connect()
		{
			return new OperaDbConnection(DBServer, DBUserName, DBPassword);
		}
        private static readonly string SySalEnvironmentKey = "_SYSAL_SK_";
		/// <summary>
		/// Creates a new set of credentials or loads credentials previously stored.
		/// </summary>
		/// <returns>The set of credentials read from the record.</returns>
		public static OperaDbCredentials CreateFromRecord()
		{
			OperaDbCredentials o = new OperaDbCredentials();            
			try
			{
                try
                {                    
                    string [] lines = System.Environment.GetEnvironmentVariable(SySalEnvironmentKey).Split(' ');
                    o.DBServer = Decode(lines[0]);
                    o.DBUserName = Decode(lines[1]);
                    o.DBPassword = Decode(lines[2]);
                    o.OPERAUserName = Decode(lines[3]);
                    o.OPERAPassword = Decode(lines[4]);
                }
                catch (Exception)
                {
                    string[] lines = System.IO.File.ReadAllLines(Path);
                    o.DBServer = Decode(lines[0]);
                    o.DBUserName = Decode(lines[1]);
                    o.DBPassword = Decode(lines[2]);
                    o.OPERAUserName = Decode(lines[3]);
                    o.OPERAPassword = Decode(lines[4]);
                }
			}
			catch (Exception)
			{
				return new OperaDbCredentials();
			}
			return o;
		}
	}

}
