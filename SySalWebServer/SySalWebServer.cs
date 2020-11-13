using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Net.Security;

namespace SySal.Web
{
    /// <summary>
    /// A browser session.
    /// </summary>
    [Serializable]
    public class Session
    {
        /// <summary>
        /// The IP address of the session client.
        /// </summary>
        public System.Net.IPAddress ClientAddress;
        /// <summary>
        /// The session cookie.
        /// </summary>
        public string Cookie;
        /// <summary>
        /// The start time for this session.
        /// </summary>
        public System.DateTime Start;
        /// <summary>
        /// The start time for this session.
        /// </summary>
        public System.DateTime End;
        /// <summary>
        /// User data.
        /// </summary>
        public object UserData;
        /// <summary>
        /// Builds a new session.
        /// </summary>
        /// <param name="clientaddr">the IP address of the client.</param>
        /// <param name="secondsduration">the time interval in seconds for the session to expire.</param>
        public Session(System.Net.IPAddress clientaddr, int secondsduration)
        {
            ClientAddress = clientaddr;
            Cookie = "";
            Start = System.DateTime.Now;
            End = Start;
            End = End.AddSeconds(secondsduration);
        }
        /// <summary>
        /// Keeps the connection alive for the specified amount of time.
        /// </summary>
        /// <param name="secondsduration">the number of seconds to add to the lifetime (starting from the renewal time).</param>
        public void KeepAlive(int secondsduration)
        {
            var end = System.DateTime.Now.AddSeconds(secondsduration);
            if (End < end) End = end;
        }
        /// <summary>
        /// Checks whether the session has expired.
        /// </summary>
        public bool Expired { get { return System.DateTime.Now >= End; } }
        /// <summary>
        /// Initializes the session cookie to a new value.
        /// </summary>
        public void InitCookie()
        {
            Cookie = System.Guid.NewGuid().ToString();
        }
    }

    /// <summary>
    /// A traceable process method request.
    /// </summary>
    [Serializable]
    public class ProcessMethodRequest
    {
        /// <summary>
        /// Client address.
        /// </summary>
        public string IPAddress = "";
        /// <summary>
        /// Headers (the first includes the page).
        /// </summary>
        public string[] Headers = new string[0];
        /// <summary>
        /// Start time when the request was issued.
        /// </summary>
        public DateTime StartTime = DateTime.Now;
    }

    /// <summary>
    /// Generic HTTP header.
    /// </summary>
    public abstract class HTTPHeader
    {
        /// <summary>
        /// Translates this HTTP header into a string suitable for HTTP transmission.
        /// </summary>
        /// <returns>the HTTP string corresponding to the name and value for this header.</returns>
        public abstract string Render();
    }

    namespace HTTPHeaders
    {
        /// <summary>
        /// "Date" HTTP header.
        /// </summary>
        public class HTTPDateHdr : HTTPHeader
        {
            public DateTime Timestamp;
            public override string Render()
            {
                return "Date: " + Timestamp.ToString("R");
            }
            public HTTPDateHdr() {}
            public HTTPDateHdr(DateTime d) { Timestamp = d; }
        }

        /// <summary>
        /// "Expires" HTTP header.
        /// </summary>
        public class HTTPExpiresHdr : HTTPHeader
        {
            public DateTime Timestamp;
            public override string Render()
            {
                return "Expires: " + Timestamp.ToString("R");
            }
            public HTTPExpiresHdr() {}
            public HTTPExpiresHdr(DateTime d) { Timestamp = d; }
        }

        /// <summary>
        /// "X-Frame-Option" HTTP header
        /// </summary>
        public class HTTPXFrameOptions : HTTPHeader
        {
            public enum HTTPFrameOption : int { SameOrigin, Deny, AllowFrom };
            public HTTPFrameOption Option = HTTPFrameOption.Deny;
            public string AllowFromOrigin = "";
            public override string Render()
            {
                string r = "X-Frame-Options: ";
                switch (Option)
                {
                    case HTTPFrameOption.SameOrigin: return r + "SAMEORIGIN";
                    case HTTPFrameOption.Deny: return r + "DENY";
                    case HTTPFrameOption.AllowFrom: return r + "ALLOW-FROM " + AllowFromOrigin;
                }
                return r;
            }
            public HTTPXFrameOptions() {}
            public HTTPXFrameOptions(HTTPFrameOption fo, string origin) { Option = fo; AllowFromOrigin = origin; }
        }
    }
    /// <summary>
    /// A response that can be split in several chunks. Must be derived to provide actual implementations of chunked transfers.
    /// </summary>        
    [Serializable]
    public abstract class ChunkedResponse
    {
        /// <summary>
        /// The chunk buffer.
        /// </summary>
        public byte[] Chunk;
        /// <summary>
        /// Remaining length of the chunk to be read. The Web automatically decreases this by the amount of valid bytes upon sending the response chunk.
        /// </summary>
        public long RemainingLength;
        /// <summary>
        /// The MIME type of the response.
        /// </summary>
        public string MimeType = "text/html; charset=utf-8";
        /// <summary>
        /// The method to be called to pump bytes into the chunk buffer.
        /// </summary>
        /// <returns>the number of filled bytes in the chunk buffer.</returns>
        public abstract long PumpBytes();
        /// <summary>
        /// Builds a new ChunkedResponse with the specified total length and chunk size.
        /// </summary>
        /// <param name="chunksize">the size of the chunk buffer to be allocated.</param>
        /// <param name="totallength">the total length of the data transfer.</param>
        public ChunkedResponse(int chunksize, long totallength)
        {
            Chunk = new byte[chunksize];
            RemainingLength = totallength;
        }
        /// <summary>
        /// HTTP headers for the response.
        /// </summary>
        public HTTPHeader[] HTTPHeaders = new HTTPHeader[0];
    }

    /// <summary>
    /// Single-chunk response for simple implementations.
    /// </summary>
    [Serializable]
    public class SingleChunkResponse : ChunkedResponse
    {
        /// <summary>
        /// Builds a single-chunk response.
        /// </summary>
        /// <param name="responsedata">the response data to be sent.</param>
        public SingleChunkResponse(byte[] responsedata)
            : base(responsedata.Length, responsedata.Length)
        {
            responsedata.CopyTo(Chunk, 0);
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(System.DateTime.Now) };
        }
        /// <summary>
        /// Pumps bytes into the chunk buffer.
        /// </summary>
        /// <returns>the size of the response.</returns>
        public override long PumpBytes()
        {
            var ret = RemainingLength;
            RemainingLength = 0;
            return ret;
        }
    }

    /// <summary>
    /// Single-chunk, HTML-only response for simple implementations.
    /// </summary>
    [Serializable]
    public class HTMLResponse : SingleChunkResponse
    {
        /// <summary>
        /// Builds a single-chunk response from an HTML string.
        /// </summary>
        /// <param name="html">the HTML string to send.</param>
        public HTMLResponse(string html) : base(Encoding.ASCII.GetBytes(html)) 
        { 
            DateTime d = System.DateTime.Now;
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(d), new HTTPHeaders.HTTPExpiresHdr(d), new HTTPHeaders.HTTPXFrameOptions(SySal.Web.HTTPHeaders.HTTPXFrameOptions.HTTPFrameOption.SameOrigin, "") };
        }
    }

    /// <summary>
    /// Sends an HTTP "Redirect" message.
    /// </summary>
    [Serializable]
    public class RedirectResponse : HTMLResponse
    {
        /// <summary>
        /// HTTP redirection code.
        /// </summary>
        public enum RedirectCode
        {             
            /// <summary>
            /// Code 303: query the new URL with a GET method (POSTed data have already been received).
            /// </summary>
            SeeOther = 303,
            /// <summary>
            /// Code 307: query the new URL with the same method.
            /// </summary>
            TemporaryRedirect = 307
        }
        /// <summary>
        /// Returns the header lines corresponding to this redirection command.
        /// </summary>
        /// <returns>the header lines to perform redirection.</returns>
        public string CodeToHeader
        {
            get
            {
                switch (Code)
                {
                    case RedirectCode.SeeOther: return ((int)Code).ToString() + " See Other\r\nLocation: " + URL; break;
                    case RedirectCode.TemporaryRedirect: return ((int)Code).ToString() + " Temporary Redirect\r\nLocation: " + URL; break;
                    default: throw new Exception("Unsupported redirection code " + Code);
                }
            }
        }
        /// <summary>
        /// The new URL to be queried.
        /// </summary>
        public string URL;
        /// <summary>
        /// The HTTP redirection code.
        /// </summary>
        public RedirectCode Code;
        /// <summary>
        /// Builds a redirection response.
        /// </summary>
        /// <param name="html">HTML code to be shown during redirection.</param>
        /// <param name="code">HTTP redirection code to be issued.</param>
        /// <param name="newurl">the new URL to be accessed.</param>
        public RedirectResponse(string html, RedirectResponse.RedirectCode code, string newurl)
            : base(html)
        {
            Code = code;
            URL = newurl;
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(System.DateTime.Now) };
        }
    }

    /// <summary>
    /// Sends an HTTP "Client Error" message.
    /// </summary>
    [Serializable]
    public class ClientErrorResponse : HTMLResponse
    {
        /// <summary>
        /// HTTP client error code.
        /// </summary>
        public enum ErrorCode
        {
            /// <summary>
            /// The request cannot be fulfilled due to bad syntax.
            /// </summary>
            BadRequest = 400,
            /// <summary>
            /// The request was a legal request, but the server is refusing to respond to it. For use when authentication is possible but has failed or not yet been provided.
            /// </summary>
            Unauthorized = 401,
            /// <summary>
            /// Reserved for future use
            /// </summary>
            PaymentRequired = 402,
            /// <summary>
            /// The request was a legal request, but the server is refusing to respond to it
            /// </summary>
            Forbidden = 403,
            /// <summary>
            /// The requested page could not be found but may be available again in the future
            /// </summary>
            NotFound = 404,
            /// <summary>
            /// A request was made of a page using a request method not supported by that page
            /// </summary>
            MethodNotAllowed = 405,
            /// <summary>
            /// The server can only generate a response that is not accepted by the client
            /// </summary>
            NotAcceptable = 406,
            /// <summary>
            /// The client must first authenticate itself with the proxy 
            /// </summary>
            ProxyAuthenticationRequired = 407,
            /// <summary>
            /// The server timed out waiting for the request 
            /// </summary>
            RequestTimeout = 408,
            /// <summary>
            /// The request could not be completed because of a conflict in the request 
            /// </summary>
            Conflict = 409,
            /// <summary>
            /// The requested page is no longer available
            /// </summary>
            Gone = 410,
            /// <summary>
            /// The "Content-Length" is not defined. The server will not accept the request without it  
            /// </summary>
            LengthRequired = 411,
            /// <summary>
            /// The precondition given in the request evaluated to false by the server  
            /// </summary>
            PreconditionFailed = 412,
            /// <summary>
            /// The server will not accept the request, because the request entity is too large   
            /// </summary>
            RequestEntityTooLarge = 413,
            /// <summary>
            /// The server will not accept the request, because the URL is too long  
            /// </summary>
            RequestURITooLong = 414,
            /// <summary>
            /// The server will not accept the request, because the media type is not supported  
            /// </summary>
            UnsupportedMediaType = 415,
            /// <summary>
            /// The client has asked for a portion of the file, but the server cannot supply that portion  
            /// </summary>
            RequestedRangeNotSatisfiable = 416,
            /// <summary>
            /// The server cannot meet the requirements of the Expect request-header field  
            /// </summary>
            ExpectationFailed = 417
        }
        /// <summary>
        /// Returns the header explanation code corresponding to this error command.
        /// </summary>
        /// <returns>the header explanation code for the error.</returns>
        public string CodeToHeader
        {
            get
            {
                switch (Code)
                {
                    case ErrorCode.BadRequest: return ((int)Code).ToString() + " Bad Request";
                    case ErrorCode.Unauthorized: return ((int)Code).ToString() + " Unauthorized";
                    case ErrorCode.PaymentRequired: return ((int)Code).ToString() + " Payment Required";
                    case ErrorCode.Forbidden: return ((int)Code).ToString() + " Forbidden";
                    case ErrorCode.NotFound: return ((int)Code).ToString() + " Not Found";
                    case ErrorCode.MethodNotAllowed: return ((int)Code).ToString() + " Method Not Allowed";
                    case ErrorCode.NotAcceptable: return ((int)Code).ToString() + " Not Acceptable";
                    case ErrorCode.ProxyAuthenticationRequired: return ((int)Code).ToString() + " Proxy Authentication Required";
                    case ErrorCode.RequestTimeout: return ((int)Code).ToString() + " Request Timeout";
                    case ErrorCode.Conflict: return ((int)Code).ToString() + " Conflict";
                    case ErrorCode.Gone: return ((int)Code).ToString() + " Gone";
                    case ErrorCode.LengthRequired: return ((int)Code).ToString() + " Length Required";
                    case ErrorCode.PreconditionFailed: return ((int)Code).ToString() + " Precondition Failed";
                    case ErrorCode.RequestEntityTooLarge: return ((int)Code).ToString() + " Request Entity Too Large";
                    case ErrorCode.RequestURITooLong: return ((int)Code).ToString() + " Request-URI Too Long";
                    case ErrorCode.UnsupportedMediaType: return ((int)Code).ToString() + " Unsupported Media Type";
                    case ErrorCode.RequestedRangeNotSatisfiable: return ((int)Code).ToString() + " Requested Range Not Satisfiable";
                    case ErrorCode.ExpectationFailed: return ((int)Code).ToString() + " Expectation Failed";
                    default: throw new Exception("Unsupported client error code " + Code);
                }
            }
        }
        /// <summary>
        /// The HTTP error code.
        /// </summary>
        public ErrorCode Code;
        /// <summary>
        /// Builds a client error response.
        /// </summary>
        /// <param name="html">HTML code to be shown for client error.</param>
        /// <param name="code">HTTP redirection code to be issued.</param>        
        public ClientErrorResponse(string html, ClientErrorResponse.ErrorCode code)
            : base(html)
        {
            Code = code;
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(System.DateTime.Now) };
        }
    }

    /// <summary>
    /// Sends an HTTP "Server Error" message.
    /// </summary>
    [Serializable]
    public class ServerErrorResponse : HTMLResponse
    {
        /// <summary>
        /// HTTP server error code.
        /// </summary>
        public enum ErrorCode
        {
            /// <summary>
            /// A generic error message, given when no more specific message is suitable
            /// </summary>
            InternalServerError = 500,
            /// <summary>
            /// The server either does not recognize the request method, or it lacks the ability to fulfill the request
            /// </summary>
            NotImplemented = 501,
            /// <summary>
            /// The server was acting as a gateway or proxy and received an invalid response from the upstream server
            /// </summary>
            BadGateway = 502,
            /// <summary>
            /// The server is currently unavailable (overloaded or down)
            /// </summary>
            ServiceUnavailable = 503,
            /// <summary>
            /// The server was acting as a gateway or proxy and did not receive a timely response from the upstream server
            /// </summary>
            GatewayTimeout = 504,
            /// <summary>
            /// The server does not support the HTTP protocol version used in the request
            /// </summary>
            HTTPVersionNotSupported = 505,
            /// <summary>
            /// The client needs to authenticate to gain network access
            /// </summary>
            NetworkAuthenticationRequired = 511
        }
        /// <summary>
        /// Returns the header explanation code corresponding to this error command.
        /// </summary>
        /// <returns>the header explanation code for the error.</returns>
        public string CodeToHeader
        {
            get
            {
                switch (Code)
                {
                    case ErrorCode.InternalServerError: return ((int)Code).ToString() + " Internal Server Error";
                    case ErrorCode.NotImplemented: return ((int)Code).ToString() + " Not Implemented";
                    case ErrorCode.BadGateway: return ((int)Code).ToString() + " Bad Gateway";
                    case ErrorCode.ServiceUnavailable: return ((int)Code).ToString() + " Service Unavailable";
                    case ErrorCode.GatewayTimeout: return ((int)Code).ToString() + " Gateway Timeout";
                    case ErrorCode.HTTPVersionNotSupported: return ((int)Code).ToString() + " HTTP Version Not Supported";
                    case ErrorCode.NetworkAuthenticationRequired: return ((int)Code).ToString() + " Network Authentication Required";
                    default: throw new Exception("Unsupported server error code " + Code);
                }
            }
        }
        /// <summary>
        /// The HTTP error code.
        /// </summary>
        public ErrorCode Code;
        /// <summary>
        /// Builds a server error response.
        /// </summary>
        /// <param name="html">HTML code to be shown for error.</param>
        /// <param name="code">HTTP redirection code to be issued.</param>
        /// <param name="newurl">the new URL to be accessed.</param>
        public ServerErrorResponse(string html, ServerErrorResponse.ErrorCode code)
            : base(html)
        {
            Code = code;
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(System.DateTime.Now) };
        }
    }

    /// <summary>
    /// Multi-chunk file transfer response.
    /// </summary>    
    [Serializable]
    public class BinaryFileResponse : ChunkedResponse
    {
        /// <summary>
        /// The path of the file to be read.
        /// </summary>
        protected string FilePath;
        
        [NonSerialized]
        private System.IO.BinaryReader ReadStream;
        /// <summary>
        /// Builds a new response, reading a specified stream.
        /// </summary>
        /// <param name="chunksize">the size of the chunk transfer buffer.</param>
        /// <param name="path">the file from which data are to be read.</param>
        /// <param name="mimetype">the MIME type of the data.</param>
        public BinaryFileResponse(int chunksize, string path, string mimetype)
            : base(chunksize, new System.IO.FileInfo(path).Length)
        {
            FilePath = path;
            ReadStream = null;
            MimeType = mimetype;
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(System.DateTime.Now) };
        }
        /// <summary>
        /// Pumps bytes into the chunk buffer.
        /// </summary>
        /// <returns>the size of the response.</returns>
        public override long PumpBytes()
        {
            if (ReadStream == null)
                ReadStream = new System.IO.BinaryReader(new System.IO.FileStream(FilePath, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read));
            long r = (long)ReadStream.Read(Chunk, 0, Chunk.Length);
            if (ReadStream.BaseStream.Position == ReadStream.BaseStream.Length)
            {
                ReadStream.BaseStream.Close();
                ReadStream = null;
            }
            return r;
        }

        ~BinaryFileResponse()
        {
            if (ReadStream != null) ReadStream.Close();
        }
    }

    /// <summary>
    /// Multi-chunk memory stream response.
    /// </summary>    
    [Serializable]
    public class ByteArrayResponse : ChunkedResponse
    {
        /// <summary>
        /// The data to be read.
        /// </summary>
        protected byte [] Data;

        [NonSerialized]
        private System.IO.BinaryReader ReadStream;
        /// <summary>
        /// Builds a new response, reading a specified stream.
        /// </summary>
        /// <param name="chunksize">the size of the chunk transfer buffer.</param>
        /// <param name="rstr">the stream from which data are to be read.</param>
        /// <param name="mimetype">the MIME type of the data.</param>
        public ByteArrayResponse(int chunksize, byte [] data, string mimetype)
            : base(chunksize, data.Length)
        {
            Data = data;
            ReadStream = null;
            MimeType = mimetype;
            DateTime d = DateTime.Now;
            HTTPHeaders = new HTTPHeader[] { new HTTPHeaders.HTTPDateHdr(d), new HTTPHeaders.HTTPExpiresHdr(d) };
        }
        /// <summary>
        /// Pumps bytes into the chunk buffer.
        /// </summary>
        /// <returns>the size of the response.</returns>
        public override long PumpBytes()
        {
            if (ReadStream == null)
                ReadStream = new System.IO.BinaryReader(new System.IO.MemoryStream(Data));
            return (long)ReadStream.Read(Chunk, 0, Chunk.Length);
        }

        ~ByteArrayResponse()
        {
            if (ReadStream != null) ReadStream.Close();
        }
    }

    /// <summary>
    /// Applications implement this interface to react on HTTP requests.
    /// </summary>
    public interface IWebApplication
    {
        /// <summary>
        /// The name of the server application.
        /// </summary>
        string ApplicationName { get; }
        /// <summary>
        /// Reacts on the HTTP GET command.
        /// </summary>
        /// <param name="sess">the session in which the call is performed.</param>
        /// <param name="page">the page that is being requested.</param>
        /// <param name="queryget">parameters in the GET method.</param>
        /// <returns>the HTTP response (most commonly an HTML page).</returns>
        ChunkedResponse HttpGet(Session sess, string page, params string[] queryget);
        /// <summary>
        /// Reacts on the HTTP POST command.
        /// </summary>
        /// <param name="sess">the session in which the call is performed.</param>
        /// <param name="page">the page that is being requested.</param>
        /// <param name="queryget">form fields sent in the method.</param>
        /// <returns>the HTTP response (most commonly an HTML page).</returns>
        ChunkedResponse HttpPost(Session sess, string page, params string[] postfields);
        /// <summary>
        /// Defines whether the Web Server should show exceptions or a generic error page;
        /// </summary>
        bool ShowExceptions { get; }
    }

    /// <summary>
    /// Extends IWebApplication with more detailed HTTP handling.
    /// </summary>
    public interface IWebApplication2 : IWebApplication
    {
        /// <summary>
        /// Retrieves the maximum data size an application expects to handle for a specific page in POST methods.
        /// </summary>
        /// <param name="sess">the current working session.</param>
        /// <param name="page">the page to be accessed.</param>
        /// <returns>the maximum number of bytes.</returns>
        /// <remarks>This overrides the setting in <see cref="SySal.Web.WebServer.m_MaxPostUrlEncodedInputLength"/>. 
        /// An application that expects to handle large amounts of data may be exposed to attackers that start sending huge files.
        /// This ensures that only some pages get large inputs, and only upon certain conditions.</remarks>
        int MaxBytesInPOSTRequest(Session sess, string page);
    }

    /// <summary>
    /// Extends IWebApplication with OPTIONS method support.
    /// </summary>
    public interface IWebApplicationOPTIONS : IWebApplication
    {
        /// <summary>
        /// Reacts on the HTTP OPTIONS command.
        /// </summary>
        /// <param name="sess">the session in which the call is performed.</param>
        /// <param name="page">the page that is being requested.</param>
        /// <param name="queryget">parameters in the GET method.</param>
        /// <returns>the HTTP response should be an empty string; headers matter.</returns>
        ChunkedResponse HttpOptions(Session sess, string page, params string[] queryget);
    }

    /// <summary>
    /// Simple application that dumps diagnostic information to a stream (e.g. console or file).
    /// </summary>
    public class DiagnosticApp : IWebApplication
    {
        /// <summary>
        /// The output stream.
        /// </summary>
        protected System.IO.TextWriter m_OutStream;

        /// <summary>
        /// Builds a new diagnostic object.
        /// </summary>
        /// <param name="tw">the stream to use to dump diagnostic data.</param>
        public DiagnosticApp(System.IO.TextWriter tw)
        {
            m_OutStream = tw;
        }

        #region IWebApplication Members

        /// <summary>
        /// The name of the application.
        /// </summary>
        public string ApplicationName
        {
            get { return GetType().ToString(); }
        }

        /// <summary>
        /// Handles Get requests.
        /// </summary>
        /// <param name="sess">the application session.</param>
        /// <param name="page">the page requested.</param>
        /// <param name="queryget">the list of parameters passed.</param>
        /// <returns>a simple web page.</returns>
        public ChunkedResponse HttpGet(Session sess, string page, params string[] queryget)
        {
            m_OutStream.WriteLine("--HttpGet--");
            m_OutStream.WriteLine((sess == null) ? "Null session" : ("Session Start: " + sess.Start + " Finish: " + sess.End + " Cookie: " + sess.Cookie + " Userdata: " + (sess.UserData == null ? "NULL" : sess.UserData.ToString())));
            m_OutStream.WriteLine("Page: " + page);
            foreach (string s in queryget)
                m_OutStream.WriteLine("QueryGet: " + s);
            m_OutStream.WriteLine("--HttpGet--");
            return new HTMLResponse("<html><head><title>Diagnostics</title></head><body>Diagnostic page - GET</body></html>\r\n");
        }

        /// <summary>
        /// Handles Post requests.
        /// </summary>
        /// <param name="sess">the application session.</param>
        /// <param name="page">the page requested.</param>
        /// <param name="postfields">the list of parameters passed.</param>
        /// <returns>a simple web page.</returns>
        public ChunkedResponse HttpPost(Session sess, string page, params string[] postfields)
        {
            m_OutStream.WriteLine("--HttpPost--");
            m_OutStream.WriteLine((sess == null) ? "Null session" : ("Session Start: " + sess.Start + " Finish: " + sess.End + " Cookie: " + sess.Cookie + " Userdata: " + (sess.UserData == null ? "NULL" : sess.UserData.ToString())));
            m_OutStream.WriteLine("Page: " + page);
            foreach (string s in postfields)
                m_OutStream.WriteLine("PostField: " + s);
            m_OutStream.WriteLine("--HttpPost--");
            return new HTMLResponse("<html><head><title>Diagnostics</title></head><body>Diagnostic page - POST</body></html>\r\n");
        }

        /// <summary>
        /// Always show exceptions.
        /// </summary>
        public bool ShowExceptions { get { return true; } }

        #endregion
    }

    /// <summary>
    /// Minimal Web server that receives HTTP queries, routes the requests to the application, and posts the result.
    /// </summary>
    public class WebServer : IDisposable
    {
        // State object for reading client data asynchronously
        internal class StateObject
        {
            // Creation time of the socket.
            public readonly System.DateTime Created = System.DateTime.Now;
            // Client socket.
            public Socket workSocket = null;
            // Size of receive buffer.
            public int BufferSize;
            // Receive buffer.
            public byte[] buffer;
            // Received data string.
            public StringBuilder sb = new StringBuilder();
            /// <summary>
            /// Builds a new StateObject with the specified buffer size.
            /// </summary>
            /// <param name="buffsize">the size of the buffer to create.</param>
            public StateObject(uint buffsize)
            {
                BufferSize = (int)buffsize;
                buffer = new byte[buffsize];
            }
        }

        /// <summary>
        /// The available sockets, one for each network interface.
        /// </summary>
        protected System.Net.Sockets.Socket[] m_LSockets;

        /// <summary>
        /// Called when a socket receives a connection request.
        /// </summary>
        /// <param name="ar">the asynchronous state.</param>
        protected void AcceptCallback(IAsyncResult ar)
        {
            Socket listener = (Socket)ar.AsyncState;
            lock (s_Waiter)
                try
                {
                    StateObject state = new StateObject(m_MaxPostUrlEncodedInputLength);
                    Socket handler = listener.EndAccept(ar);
                    LingerOption lo = new LingerOption(true, m_SecondsDuration);
                    state.workSocket = handler;
                    System.Threading.Thread thr = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(ReadHTTPRequest));
                    thr.Start(state);
                }
                catch (Exception ax)
                {
                    if (ResponseLog != null)
                        try
                        {
                            ResponseLog.WriteLine(System.DateTime.Now + ": " + ax.ToString());
                        }
                        catch (Exception) { }
                }
                finally
                {
                    listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
                }
        }


        /// <summary>
        /// List of temporary files that are in use and should not be deleted.
        /// </summary>
        protected List<string> TempFiles = new List<string>();

        /// <summary>
        /// Table of active sessions.
        /// </summary>
        protected System.Collections.Specialized.OrderedDictionary ActiveConnections = new System.Collections.Specialized.OrderedDictionary();

        /// <summary>
        /// Retrieves the count of active connections.
        /// </summary>
        public int Connections { get { return ActiveConnections.Count; } }

        /// <summary>
        /// Event handler for <see cref="OnBeforeConnection"/>.
        /// </summary>
        /// <param name="ws">the WebServer that is going to accept the connection.</param>
        public delegate void dBeforeConnection(WebServer ws, Session sess);

        /// <summary>
        /// Event handler for <see cref="OnAfterConnection"/>.
        /// </summary>
        /// <param name="ws">the WebServer that accepted the connection.</param>
        public delegate void dAfterConnection(WebServer ws, Session sess);

        /// <summary>
        /// Event handler for <see cref="OnBeforeDisconnection"/>.
        /// </summary>
        /// <param name="ws">the WebServer that is disconnecting a client.</param>
        public delegate void dBeforeDisconnection(WebServer ws, Session sess);

        /// <summary>
        /// Event handler for <see cref="OnAfterDisconnection"/>.
        /// </summary>
        /// <param name="ws">the WebServer that disconnected a client.</param>
        public delegate void dAfterDisconnection(WebServer ws, Session sess);

        /// <summary>
        /// Event triggered before a new connection starts.
        /// </summary>
        public event dBeforeConnection OnBeforeConnection;

        /// <summary>
        /// Event triggered after a new connection is accepted.
        /// </summary>
        public event dAfterConnection OnAfterConnection;

        /// <summary>
        /// Event triggered before a connection is terminated.
        /// </summary>
        public event dBeforeDisconnection OnBeforeDisconnection;

        /// <summary>
        /// Event triggered after a connection is terminated.
        /// </summary>
        public event dAfterDisconnection OnAfterDisconnection;

        protected List<ProcessMethodRequest> ProcessRequests = new List<ProcessMethodRequest>();

        SySal.Web.Session FindActiveSession(string cookie, IPAddress clientaddr)
        {
            string ip = clientaddr.ToString();
            string lcookie = cookie.ToLower();
            string sk = lcookie + ip;
            lock (ActiveConnections)
            {
                if (ActiveConnections.Contains(sk))
                    return (Session)ActiveConnections[sk];
                string classC = ip.Substring(0, ip.LastIndexOf('.'));
                sk = lcookie + classC + ".0";
                if (ActiveConnections.Contains(sk))
                    return (Session)ActiveConnections[sk];
                string classB = classC.Substring(0, classC.LastIndexOf('.'));
                sk = lcookie + classB + ".0.0";
                if (ActiveConnections.Contains(sk))
                    return (Session)ActiveConnections[sk];
            }
            return null;
        }

        /// <summary>
        /// The string that has been received is checked for syntax and the appropriate method is called.
        /// </summary>
        /// <param name="netstr">the network stream that encapsulates the socket.</param>
        /// <param name="sslstr">the SSL stream that encapsulates the network stresam, if SSL connection is used.</param>
        /// <param name="clientaddr">the IP address of the client.</param>
        /// <param name="headers">the HTTP headers sent.</param>
        /// <param name="timeout">timeout to complete reading.</param>
        /// <param name="worksession">the resulting working sessions.</param>
        /// <returns>the HTTP response of the application, if any.</returns>
        private ChunkedResponse ProcessMethod(NetworkStream netstr, SslStream sslstr, IPAddress clientaddr, string[] headers, System.DateTime timeout, out Session worksession)
        {
            ProcessMethodRequest pmr = new ProcessMethodRequest() { IPAddress = clientaddr.ToString(), Headers = headers };
            try
            {
                lock (ProcessRequests)
                    ProcessRequests.Add(pmr);
                System.Text.RegularExpressions.Match m = null;
                bool ishead = false;
                m = s_OptionsMethod.Match(headers[0]);
                if (m.Success && m.Index == 0)
                {
                    string page;
                    string[] args = new string[0];
                    int checkq = m.Groups[1].Value.IndexOf("?");
                    if (checkq < 0) page = m.Groups[1].Value;
                    else
                    {
                        page = m.Groups[1].Value.Substring(0, checkq);
                        args = m.Groups[1].Value.Substring(checkq + 1).Split('&');
                        if (m_AutoQueryDecode)
                        {
                            int a1;
                            for (a1 = 0; a1 < args.Length; a1++)
                                args[a1] = URLDecode(args[a1]);
                        }
                    }
                    Session sess = null;
                    foreach (string line in headers)
                    {
                        m = s_Cookie.Match(line);
                        if (m.Success && m.Index == 0)
                        {
                            string[] cks = line.Substring(m.Groups[0].Index + m.Groups[0].Length).Split(';');
                            foreach (string ck in cks)
                            {
                                m = s_CookieValue.Match(ck);
                                if (String.Compare(m.Groups[1].Value, SIDCookie, true) == 0)
                                {
                                    lock (ActiveConnections)
                                        sess = FindActiveSession(m.Groups[2].Value, clientaddr);
                                    if (sess == null || sess.Expired) sess = new Session(clientaddr, m_SecondsDuration);
                                    else sess.KeepAlive(m_SecondsDuration);
                                }
                            }
                        }
                    }
                    if (sess == null) sess = new Session(clientaddr, m_SecondsDuration);
                    worksession = sess;
                    if (m_App is IWebApplicationOPTIONS) return (m_App as IWebApplicationOPTIONS).HttpOptions(sess, page, args);
                    else return new SySal.Web.HTMLResponse("");
                }
                m = s_GetMethod.Match(headers[0]);
                if (m.Success == false || m.Index > 0)
                {
                    ishead = true;
                    m = s_HeadMethod.Match(headers[0]);
                }
                if (m.Success && m.Index == 0)
                {
                    string page;
                    string[] args = new string[0];
                    int checkq = m.Groups[1].Value.IndexOf("?");
                    if (checkq < 0) page = m.Groups[1].Value;
                    else
                    {
                        page = m.Groups[1].Value.Substring(0, checkq);
                        args = m.Groups[1].Value.Substring(checkq + 1).Split('&');
                        if (m_AutoQueryDecode)
                        {
                            int a1;
                            for (a1 = 0; a1 < args.Length; a1++)
                                args[a1] = URLDecode(args[a1]);
                        }
                    }
                    Session sess = null;
                    foreach (string line in headers)
                    {
                        m = s_Cookie.Match(line);
                        if (m.Success && m.Index == 0)
                        {
                            string[] cks = line.Substring(m.Groups[0].Index + m.Groups[0].Length).Split(';');
                            foreach (string ck in cks)
                            {
                                m = s_CookieValue.Match(ck);
                                if (String.Compare(m.Groups[1].Value, SIDCookie, true) == 0)
                                {
                                    lock (ActiveConnections)
                                        sess = FindActiveSession(m.Groups[2].Value, clientaddr);
                                    if (sess == null || sess.Expired) sess = new Session(clientaddr, m_SecondsDuration);
                                    else sess.KeepAlive(m_SecondsDuration);
                                }
                            }
                        }
                    }
                    if (sess == null) sess = new Session(clientaddr, m_SecondsDuration);
                    worksession = sess;
                    ChunkedResponse response = m_App.HttpGet(sess, page, args);
                    if (ishead) response.Chunk = null;
                    return response;
                }
                m = s_PostMethod.Match(headers[0]);
                {
                    string page;
                    string[] args = new string[0];
                    int checkq = m.Groups[1].Value.IndexOf("?");
                    if (checkq < 0) page = m.Groups[1].Value;
                    else
                    {
                        page = m.Groups[1].Value.Substring(0, checkq);
                        args = m.Groups[1].Value.Substring(checkq + 1).Split('&');
                        if (m_AutoQueryDecode)
                        {
                            int a1;
                            for (a1 = 0; a1 < args.Length; a1++)
                                args[a1] = URLDecode(args[a1]);
                        }
                    }
                    Session sess = null;
                    foreach (string line in headers)
                    {
                        m = s_Cookie.Match(line);
                        if (m.Success && m.Index == 0)
                        {
                            string[] cks = line.Substring(m.Groups[0].Index + m.Groups[0].Length).Split(';');
                            foreach (string ck in cks)
                            {
                                m = s_CookieValue.Match(ck);
                                if (String.Compare(m.Groups[1].Value, SIDCookie, true) == 0)
                                {
                                    lock (ActiveConnections)
                                        sess = FindActiveSession(m.Groups[2].Value, clientaddr);
                                    if (sess == null || sess.Expired) sess = new Session(clientaddr, m_SecondsDuration);
                                    else sess.KeepAlive(m_SecondsDuration);
                                }
                            }
                        }
                    }
                    if (sess == null) sess = new Session(clientaddr, m_SecondsDuration);
                    worksession = sess;
                    int contentlength = -1;
                    string multipartboundary = "";
                    ContentType contenttype = ContentType.None;
                    foreach (string line in headers)
                    {
                        m = s_ContentType.Match(line);
                        if (m.Success && m.Index == 0)
                        {
                            if (String.Compare(m.Groups[1].Value, ContentTypeString(ContentType.ApplicationXWWWFormUrlEncoded), true) == 0)
                                contenttype = ContentType.ApplicationXWWWFormUrlEncoded;
                            else if (String.Compare(m.Groups[1].Value, ContentTypeString(ContentType.MultipartFormData), true) == 0)
                            {
                                contenttype = ContentType.MultipartFormData;
                                int n = m.Index + m.Length;
                                m = s_MultipartBoundary.Match(line, m.Index + m.Length);
                                if (m.Success && m.Index >= n)
                                    multipartboundary = m.Groups[1].Value;
                                else throw new Exception("Can't find multipart boundary.");
                            }
                            else throw new Exception("Unsupported encoding \"" + m.Groups[1].Value + "\".");
                        }
                        m = s_ContentLength.Match(line);
                        if (m.Success && m.Index == 0)
                        {
                            contentlength = Convert.ToInt32(m.Groups[1].Value);
                            if (m_App is IWebApplication2) contentlength = Math.Min(contentlength, (m_App as IWebApplication2).MaxBytesInPOSTRequest(sess, page));
                        }
                    }
                    string[] postfields = null;
                    System.IO.FileStream[] lockstreams = null;
                    try
                    {
                        switch (contenttype)
                        {
                            case ContentType.ApplicationXWWWFormUrlEncoded: postfields = ReadHTTPUrlEncodedPostFields(netstr, sslstr, contentlength, timeout); break;
                            case ContentType.MultipartFormData: postfields = ReadHTTPMultipartFormDataPostFields(netstr, sslstr, "\r\n--" + multipartboundary, contentlength, timeout, out lockstreams); break;
                        }
                        if (postfields != null)
                        {
                            string[] allpostfields = new string[args.Length + postfields.Length];
                            args.CopyTo(allpostfields, 0);
                            postfields.CopyTo(allpostfields, args.Length);
                            args = allpostfields;
                        }
                        return m_App.HttpPost(sess, page, args);
                    }
                    finally
                    {
                        if (lockstreams != null)
                            foreach (System.IO.FileStream sm in lockstreams)
                            {
                                sm.Close();
                                lock (TempFiles)
                                    TempFiles.RemoveAll(a => string.Compare(a, sm.Name, true) == 0);                                        
                            }
                    }
                }
                throw new Exception("Unsupported method.");
            }
            finally
            {
                lock (ProcessRequests)
                    try
                    {
                        ProcessRequests.Remove(pmr);
                    }
                    catch (Exception) { }
            }
        }

        string[] ReadHTTPUrlEncodedPostFields(NetworkStream netstr, SslStream sslstr, int contentlength, System.DateTime timeout)
        {
            if (contentlength > 0 && contentlength <= m_MaxPostUrlEncodedInputLength)
            {
                byte[] b = new byte[contentlength];
                int br = 0;
                if (sslstr != null)
                    while (br < contentlength && System.DateTime.Now <= timeout)
                        br += sslstr.Read(b, br, b.Length - br);
                else
                    while (br < contentlength && System.DateTime.Now <= timeout)
                        br += netstr.Read(b, br, b.Length - br);
                var ret = System.Text.Encoding.ASCII.GetString(b).Split('&');
                if (m_AutoQueryDecode)
                {
                    int i;
                    for (i = 0; i < ret.Length; i++)
                        ret[i] = URLDecode(ret[i]);
                }
                return ret;
            }
            return null;
        }

        int TimedRead(NetworkStream ns, byte[] buffer)
        {
            int br = 0;
            int blen = buffer.Length;
            System.DateTime timeout = System.DateTime.Now;
            timeout.AddMilliseconds(m_SocketTimeout);
            while (br < blen && System.DateTime.Now <= timeout)
                br += ns.Read(buffer, br, blen - br);
            if (br < blen) throw new Exception("Incomplete read.");
            return br;
        }

        string[] ReadHTTPHeaders(NetworkStream nns, SslStream sslns, ref int bytesread, int contentlength, string client, System.DateTime created)
        {
            try
            {
                int timeout = 5000;
                if (sslns == null)
                    while (nns.DataAvailable == false)
                    {
                        if (timeout < 0) throw new Exception("ReadHTTPHeaders wait timeout.");
                        System.Threading.Thread.Sleep(100);
                        timeout -= 100;
                    }
                byte[] b = new byte[m_MaxPostUrlEncodedInputLength];
                byte t0, t1, t2;
                System.IO.Stream ns = sslns;
                if (ns == null) ns = nns;
                t0 = b[0] = (byte)ns.ReadByte();
                VerboseLogSocketDirLog(client, created, Encoding.ASCII.GetString(new byte[] { t0 }) + " " + t0.ToString() + "\r\n");
                t1 = b[1] = (byte)ns.ReadByte();
                VerboseLogSocketDirLog(client, created, Encoding.ASCII.GetString(new byte[] { t1 }) + " " + t1.ToString() + "\r\n");
                int br = 2;
                bytesread += br;
                bool terminatorfound = false;
                while (br < b.Length && bytesread < contentlength)
                {
                    if (t0 == 10)
                    {
                        if (t1 == t0)
                        {
                            terminatorfound = true;
                            break;
                        }
                        t2 = (byte)ns.ReadByte();
                        VerboseLogSocketDirLog(client, created, Encoding.ASCII.GetString(new byte[] { t2 }) + " " + t2.ToString() + "\r\n");
                        bytesread++;
                        b[br++] = t2;
                        if (t2 == 10 && t1 == 13)
                        {
                            terminatorfound = true;
                            break;
                        }
                    }
                    else
                    {
                        b[br++] = t2 = (byte)ns.ReadByte();
                        VerboseLogSocketDirLog(client, created, Encoding.ASCII.GetString(new byte[] { t2 }) + " " + t2.ToString() + "\r\n");
                        bytesread++;
                    }
                    t0 = t1;
                    t1 = t2;
                }
                if (terminatorfound == false) return new string[0];
                return Encoding.ASCII.GetString(b, 0, br).Split(new string[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);
            }
            catch (Exception x)
            {
                if (ResponseLog != null)
                    try
                    {
                        ResponseLog.WriteLine(x.ToString());
                        ResponseLog.WriteLine("Thread " + System.Threading.Thread.CurrentThread.ManagedThreadId + " Bytes read: " + bytesread + " Contentlength: " + contentlength);
                    }
                    catch (Exception) { }
                //Console.WriteLine(x.ToString());
                //Console.WriteLine("Thread " + System.Threading.Thread.CurrentThread.ManagedThreadId + " Bytes read: " + bytesread + " Contentlength: " + contentlength);
                throw x;
            }
        }

        string[] ReadHTTPMultipartFormDataPostFields(NetworkStream netstr, SslStream sslstr, string boundary, int contentlength, System.DateTime timeout, out System.IO.FileStream[] lockstreams)
        {
            System.Collections.ArrayList a_postfields = new System.Collections.ArrayList();
            System.Collections.ArrayList a_lockstreams = new System.Collections.ArrayList();
            try
            {
                byte[] boundarybuff = System.Text.Encoding.ASCII.GetBytes(boundary);
                int boundarybufflen = boundary.Length;
                byte boundarybufflast = boundarybuff[boundarybuff.Length - 1];
                if (contentlength > 0 && contentlength <= m_MaxMultipartLength)
                {
                    int bytesread = 0;
                    int i, br;
                    byte b;
                    bool firstloop = true;
                    while (bytesread < contentlength)
                    {
                        if (firstloop)
                        {
                            byte[] b_data = new byte[boundarybufflen];
                            b_data[0] = 13;
                            b_data[1] = 10;
                            bytesread = br = 2;
                            while (bytesread < contentlength)
                            {
                                b = (byte)((sslstr != null) ? sslstr.ReadByte() : netstr.ReadByte());
                                b_data[br++ % boundarybufflen] = b;
                                bytesread++;
                                if (b == boundarybufflast && br >= boundarybufflen)
                                {
                                    for (i = 0; i < boundarybufflen; i++)
                                        if (boundarybuff[i] != b_data[(br - boundarybufflen + i) % boundarybufflen])
                                            break;
                                    if (i == boundarybufflen)
                                    {
                                        br -= boundarybufflen;
                                        break;
                                    }
                                }
                            }
                            if (bytesread >= contentlength) throw new Exception("Can't find MIME boundary.");
                        }
                        firstloop = false;
                        br = 0;
                        string[] headers = ReadHTTPHeaders(netstr, sslstr, ref bytesread, contentlength, "InMultipartRead", System.DateTime.Now);
                        System.Text.RegularExpressions.Match m = null;
                        foreach (string h in headers)
                        {
                            m = s_ContentDisposition.Match(h);
                            if (m.Success && m.Index == 0)
                                if (String.Compare(m.Groups[1].Value, "form-data", true) == 0)
                                {
                                    if (ResponseLog != null) ResponseLog.WriteLine("Content-Disposition header is : " + h);
                                    int n = m.Index + m.Length;
                                    string name = "";
                                    string filename = "";
                                    m = s_ContentDispositionName.Match(h, n);
                                    if (ResponseLog != null) ResponseLog.WriteLine("Content-Disposition name match : " + m.Success);
                                    if (m.Success == false) throw new Exception("Could not find field name.");
                                    name = m.Groups[1].Value;
                                    m = s_ContentDispositionFileName.Match(h, n);
                                    if (ResponseLog != null) ResponseLog.WriteLine("Content-Disposition filename match : " + m.Success);
                                    if (m.Success == false)
                                    {
                                        byte[] b_data = new byte[m_MaxPostUrlEncodedInputLength];
                                        br = 0;
                                        while (br < m_MaxPostUrlEncodedInputLength && bytesread < contentlength)
                                        {
                                            b = (byte)((sslstr != null) ? sslstr.ReadByte() : netstr.ReadByte());
                                            b_data[br++] = b;
                                            bytesread++;
                                            if (b == boundarybufflast && br >= boundarybufflen)
                                            {
                                                for (i = 0; i < boundarybufflen; i++)
                                                    if (boundarybuff[i] != b_data[br - boundarybufflen + i])
                                                        break;
                                                if (i == boundarybufflen)
                                                {
                                                    br -= boundarybufflen;
                                                    break;
                                                }
                                            }
                                        }
                                        string fieldvalue = System.Text.Encoding.ASCII.GetString(b_data, 0, br);
                                        a_postfields.Add(name + "=" + fieldvalue);
                                    }
                                    else
                                    {
                                        filename = m.Groups[1].Value;
                                        string fieldvalue = SentFileDirectory + "_" + System.Guid.NewGuid().ToString() + " " + filename;
                                        lock (TempFiles)
                                            TempFiles.Add(fieldvalue);
                                        System.IO.FileStream wstr = new System.IO.FileStream(fieldvalue, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.Read);
                                        byte[] b_data = new byte[boundarybufflen];
                                        br = 0;
                                        while (bytesread < contentlength)
                                        {
                                            b = (byte)((sslstr != null) ? sslstr.ReadByte() : netstr.ReadByte());
                                            b_data[br++ % boundarybufflen] = b;
                                            bytesread++;
                                            if (br >= boundarybufflen)
                                            {
                                                if (b == boundarybufflast)
                                                {
                                                    for (i = 0; i < boundarybufflen; i++)
                                                        if (boundarybuff[i] != b_data[(br - boundarybufflen + i) % boundarybufflen])
                                                            break;
                                                    if (i == boundarybufflen)
                                                    {
                                                        br -= boundarybufflen;
                                                        break;
                                                    }
                                                }
                                                wstr.WriteByte(b_data[br % boundarybufflen]);
                                            }
                                        }
                                        a_postfields.Add(name + "=" + fieldvalue);
                                        wstr.Flush();
                                        a_lockstreams.Add(wstr);
                                    }
                                }
                        }
                    }
                }
                lockstreams = (System.IO.FileStream[])a_lockstreams.ToArray(typeof(System.IO.FileStream));
                return (string[])a_postfields.ToArray(typeof(string));
            }
            catch (Exception x)
            {
                foreach (System.IO.Stream sm in a_lockstreams)
                    sm.Close();
                throw x;
            }
        }

        static object s_Waiter = new object();

        static byte[] s_ChunkedTrailerBuffer = Encoding.ASCII.GetBytes("\r\n");

        static byte[] s_ChunkEndBuffer = Encoding.ASCII.GetBytes("\r\n\r\n");

        public System.Security.Authentication.CipherAlgorithmType[] EnabledCipherAlgorithms = new System.Security.Authentication.CipherAlgorithmType[] { System.Security.Authentication.CipherAlgorithmType.Aes128, System.Security.Authentication.CipherAlgorithmType.Aes192, System.Security.Authentication.CipherAlgorithmType.Aes256 };

        public System.Security.Authentication.SslProtocols EnabledSSLProtocols = System.Security.Authentication.SslProtocols.Tls;

        static void _debugdump(string x)
        {
            System.IO.File.AppendAllText("/dev/shm/webserver", x + "\n");
        }

        private void ReadHTTPRequest(object obj)
        {
            StateObject state = obj as StateObject;
            NetworkStream netstr = null;            
            try
            {
                VerboseLogSocketDirLog(state.workSocket.RemoteEndPoint.ToString(), state.Created, "OPEN\r\n");
                int bytesread = 0;                
                Socket handler = state.workSocket;
                handler.SendTimeout = handler.ReceiveTimeout = m_SocketTimeout;
                //handler.DontFragment = true;
                handler.DontFragment = false;
                handler.Ttl = 255;
                System.DateTime timeout = System.DateTime.Now;
                timeout = timeout.AddMilliseconds(handler.ReceiveTimeout);
                handler.SendTimeout = 5000;                
                netstr = new NetworkStream(handler, false);                
                netstr.ReadTimeout = 5000;
                netstr.WriteTimeout = 5000;
                SslStream sslstr = null;
                if (X509ServerCertificate != null)
                {                                    	
                    sslstr = new SslStream(netstr, false,null,null);
                    sslstr.AuthenticateAsServer(X509ServerCertificate, false, EnabledSSLProtocols, true);
                    sslstr.ReadTimeout = 5000;
                    sslstr.WriteTimeout = 5000;
                    handler.DontFragment = false;
                    /*
					try
                    {
                    	System.IO.File.WriteAllText("/km3net/data/logs/ssl.txt", "\n" + DateTime.Now + " enabled " + ((int)EnabledSSLProtocols).ToString() + " used " + ((int)sslstr.SslProtocol).ToString() + " ciphalgo " + sslstr.CipherAlgorithm);
                    }
                    catch (Exception)
                    {

                    }
                    int e;
                    for (e = 0; e < EnabledCipherAlgorithms.Length; e++)
                        if (EnabledCipherAlgorithms[e] == sslstr.CipherAlgorithm)
                            break;
                    if (e == EnabledCipherAlgorithms.Length)                    
                    {
                    	sslstr.Close();
                    	return;
                    }*/
                }
                string [] headers = null;
                try
                {
                    headers = ReadHTTPHeaders(netstr, sslstr, ref bytesread, (int)m_MaxPostUrlEncodedInputLength, state.workSocket.RemoteEndPoint.ToString(), state.Created);
                }
                catch (Exception x)
                {
                    if (ResponseLog != null)
                        try
                        {
                            ResponseLog.WriteLine("Socket Remote " + (handler.RemoteEndPoint).ToString() + " Local " + (handler.LocalEndPoint).ToString() + "\r\n" + x.ToString());
                        }
                        catch (Exception) { }
                    throw new Exception("Cannot find HTTP headers.\r\n" + x.ToString());
                }

                Session worksession = null;
                string initcookie = "";

                bool showexceptions = false;
                string response = null;
                ChunkedResponse responsebytes = null;
                try
                {
                    try
                    {
                        if (ResponseLog != null)                        
                            foreach (string h in headers)
                                ResponseLog.WriteLine(h);
                    }
                    catch (Exception) { }
                    if (m_App != null) showexceptions = m_App.ShowExceptions;
                    responsebytes = (m_App != null) ? ProcessMethod(netstr, sslstr, ((System.Net.IPEndPoint)handler.RemoteEndPoint).Address, headers, timeout, out worksession) : null;
                }
                catch (Exception x)
                {
                    if (ResponseLog != null) ResponseLog.WriteLine(x.ToString());
                    //Console.WriteLine(x.ToString());
                    if (showexceptions) response = "<html><head><title>Exception</title></head><body><h1>Exception</h1><p><font color=\"red\">" + HtmlFormat(x.ToString()) + "</font></p></body>\r\n";
                    else response = null;
                }
                CheckExpiredConnections();
                if (responsebytes != null && worksession != null && worksession.Cookie.Length == 0)
                {
                    worksession.InitCookie();
                    lock (ActiveConnections)
                    {
                        if (OnBeforeConnection != null) OnBeforeConnection(this, worksession);
                        ActiveConnections.Add(worksession.Cookie.ToLower() + worksession.ClientAddress.ToString(), worksession);
                        if (OnAfterConnection != null) OnAfterConnection(this, worksession);
                    }
                    initcookie = "Set-Cookie: " + SIDCookie + "=" + worksession.Cookie + " ; HttpOnly" + ((sslstr != null) ? "; Secure " : "") + "\r\n";
                }

                //System.Threading.Thread.Sleep(100);

                bool chunked = false;
                if (responsebytes == null) response = "HTTP/1.0 404 Not Found\r\n" + ((m_App.ApplicationName == null) ? "" : ("Server: " + m_App.ApplicationName + "\r\n")) + "\r\nContent-Length: 0\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n<html><body>Resource not found.</body></html>\r\n\r\n";
                else
                {
                    if (responsebytes.RemainingLength < 0) chunked = true;
                    if (responsebytes is RedirectResponse)
                    {
                        response = "HTTP/1.1 " + (responsebytes as RedirectResponse).CodeToHeader + "\r\n" + ((m_App.ApplicationName == null) ? "" : ("Server: " + m_App.ApplicationName + "\r\n")) + "Date: " + System.DateTime.Now.ToString("R") + "\r\n" + initcookie + "Content-Type: " + responsebytes.MimeType + "\r\n" + (chunked ? "Transfer-Encoding: chunked\r\n" : ("Content-Length: " + responsebytes.RemainingLength + "\r\nConnection: close\r\n")) + "\r\n";
                    }
                    else if (responsebytes is ClientErrorResponse)
                    {
                        response = "HTTP/1.1 " + (responsebytes as ClientErrorResponse).CodeToHeader + "\r\n" + ((m_App.ApplicationName == null) ? "" : ("Server: " + m_App.ApplicationName + "\r\n")) + "Date: " + System.DateTime.Now.ToString("R") + "\r\n" + initcookie + "Content-Type: " + responsebytes.MimeType + "\r\n" + (chunked ? "Transfer-Encoding: chunked\r\n" : ("Content-Length: " + responsebytes.RemainingLength + "\r\nConnection: close\r\n")) + "\r\n";
                    }
                    else if (responsebytes is ServerErrorResponse)
                    {
                        response = "HTTP/1.1 " + (responsebytes as ServerErrorResponse).CodeToHeader + "\r\n" + ((m_App.ApplicationName == null) ? "" : ("Server: " + m_App.ApplicationName + "\r\n")) + "Date: " + System.DateTime.Now.ToString("R") + "\r\n" + initcookie + "Content-Type: " + responsebytes.MimeType + "\r\n" + (chunked ? "Transfer-Encoding: chunked\r\n" : ("Content-Length: " + responsebytes.RemainingLength + "\r\nConnection: close\r\n")) + "\r\n";
                    }
                    else
                    {
                        //response = "HTTP/1.1 200 OK\r\nServer: " + m_App.ApplicationName + "\r\nDate: " + System.DateTime.Now.ToString("R") + "\r\n" + initcookie + "Content-Type: " + responsebytes.MimeType + "\r\n" + (chunked ? "Transfer-Encoding: chunked\r\n" : ("Content-Length: " + responsebytes.RemainingLength + "\r\nConnection: close\r\n")) + "\r\n";
                        response = "HTTP/1.1 200 OK\r\nServer: " + m_App.ApplicationName + "\r\n" + initcookie + "Content-Type: " + responsebytes.MimeType + "\r\n" + (chunked ? "Transfer-Encoding: chunked\r\n" : ("Content-Length: " + responsebytes.RemainingLength + "\r\nConnection: close\r\n"));
                        foreach (HTTPHeader hdr in responsebytes.HTTPHeaders)
                            response += hdr.Render() + "\r\n";
                        response += "\r\n";
                    }
                }

                if (ResponseLog != null)
                    try
                    {
                        ResponseLog.WriteLine("Socket Remote " + (handler.RemoteEndPoint).ToString() + " Local " + (handler.LocalEndPoint).ToString());
                        ResponseLog.WriteLine(response);
                        ResponseLog.Flush();
                    }
                    catch (Exception) { }

                byte[] responsebytesbuffer = Encoding.ASCII.GetBytes(response);
                System.Threading.Thread.Sleep(10);
                if (sslstr != null) sslstr.Write(responsebytesbuffer, 0, responsebytesbuffer.Length);
                else netstr.Write(responsebytesbuffer, 0, responsebytesbuffer.Length);
                //                netstr.Flush();                
                int byteswritten = 0;
                foreach (var h1 in headers)
                    //_debugdump(h1)
                    ;
                //_debugdump("responsebytes NULL? " + (responsebytes == null));
                //_debugdump("responsebytes.Chunk NULL? " + ((responsebytes?.Chunk ?? null) == null));
                if (responsebytes != null && responsebytes.Chunk != null)                
                    do
                    {
                        //_debugdump("LOOP");
                        int retrycount = 3;
                        while (retrycount-- > 0)
                            try
                            {
                                //_debugdump("try retrycount " + retrycount);
                                byteswritten = (int)responsebytes.PumpBytes();
                                //_debugdump("byteswritten " + byteswritten);
                                if (byteswritten > 0)
                                    if (chunked)
                                    {
                                        //_debugdump("chunked");
                                        string chunkhdr = byteswritten.ToString("X") + "\r\n";                                    
                                        byte[] chunkhdrbuffer = Encoding.ASCII.GetBytes(chunkhdr);
                                        if (sslstr != null)
                                        {
                                            //_debugdump("SSL");
                                            sslstr.Write(chunkhdrbuffer, 0, chunkhdrbuffer.Length);
                                            if (byteswritten > 0)
                                            {
                                                sslstr.Write(responsebytes.Chunk, 0, (int)byteswritten);
                                                sslstr.Write(s_ChunkedTrailerBuffer, 0, s_ChunkedTrailerBuffer.Length);
                                            }
                                            else
                                                sslstr.Write(s_ChunkedTrailerBuffer, 0, s_ChunkedTrailerBuffer.Length);
                                            sslstr.Flush();
                                            //_debugdump("Flush");
                                        }
                                        else
                                        {
                                            //_debugdump("clear");
                                            netstr.Write(chunkhdrbuffer, 0, chunkhdrbuffer.Length);
                                            if (byteswritten > 0)
                                            {
                                                netstr.Write(responsebytes.Chunk, 0, (int)byteswritten);
                                                netstr.Write(s_ChunkedTrailerBuffer, 0, s_ChunkedTrailerBuffer.Length);
                                            }
                                            else
                                                netstr.Write(s_ChunkedTrailerBuffer, 0, s_ChunkedTrailerBuffer.Length);
                                            netstr.Flush();
                                            //_debugdump("Flush");
                                        }
                                    }
                                    else
                                    {
                                        //_debugdump("onepiece");
                                        if (sslstr != null) sslstr.Write(responsebytes.Chunk, 0, (int)byteswritten);
                                        else netstr.Write(responsebytes.Chunk, 0, (int)byteswritten);
                                    }
                                //                                netstr.Flush();
                               //_debugdump("remaininglength " + responsebytes.RemainingLength + " " + byteswritten);
                                /*if (responsebytes.RemainingLength >= 0)
                                    responsebytes.RemainingLength -= byteswritten;*/
                                //_debugdump("remaininglength " + responsebytes.RemainingLength);
                                break;
                            }
                            catch (Exception x5) 
                            {
                                //_debugdump("exception " + x5.ToString());
                                if (ResponseLog != null)
                                    try
                                    {
                                        ResponseLog.WriteLine(x5.ToString());
                                    }
                                    catch (Exception) { }
                            }
                        //_debugdump("retrycount");
                        if (retrycount <= 0)
                        {
                            if (ResponseLog != null)
                                try
                                {
                                    ResponseLog.WriteLine("No more trials to do, aborting transmission.\r\nHeader: " + headers[0] + "\r\nBytes remaining: " + responsebytes.RemainingLength);
                                }
                                catch (Exception) { }
                            break;
                        }
                        //_debugdump("ENDLOOP " + responsebytes.RemainingLength + " " + byteswritten);
                    }
                    while (responsebytes.RemainingLength != 0 || byteswritten != 0);
                //_debugdump("Final flush");
                if (sslstr != null) sslstr.Flush();
                netstr.Flush();
                try
                {
                    //_debugdump("Zero write A");
                    handler.SendTimeout = 10000;
                    //_debugdump("Zero write B");
                    handler.Send(new byte[0]);
                    //_debugdump("Zero write C");
                    System.Threading.Thread.Sleep(100);
                }
                catch (Exception x6) 
                {
                    //_debugdump("Exception " + x6.ToString());
                }
                //System.Threading.Thread.Sleep(10000);
                //_debugdump("Close");
                if (sslstr != null) sslstr.Close();
                netstr.Close();
            }
            catch (Exception rx)
            {
                //_debugdump("Exception " + rx.ToString());
                if (ResponseLog != null)
                    try
                    {
                        ResponseLog.WriteLine(System.DateTime.Now + ": " + rx.ToString());
                    }
                    catch (Exception) { }
            }
            finally
            {                
                if (netstr != null)
                {
                    netstr.Flush();
                    netstr.Close();
                    netstr.Dispose();
                }
                try
                {
                    Socket handler = state.workSocket;
                    handler.Close();
                }
                catch (Exception) { }
            }
        }

        public System.IO.TextWriter ResponseLog;

        public string SentFileDirectory;

        public string VerboseLogPathTemplate;

        public void VerboseLogSocketDirLog(string client, System.DateTime tm, string text)
        {
            if (VerboseLogPathTemplate != null)
                try
                {
                    System.IO.File.AppendAllText(
                        VerboseLogPathTemplate + "_" + client.Replace(":", "_") + "__" +
                            tm.Year + "_" + tm.Month + "_" + tm.Day + "_" + tm.Hour + "_" + tm.Minute + "_" + tm.Second + "_" + tm.Millisecond + ".log",
                            text);
                }
                catch (Exception) { }
        }

        void CheckExpiredConnections()
        {
            lock (ActiveConnections)
                if (ActiveConnections.Count > 10)
                {
                    int i;
                    for (i = 0; i < ActiveConnections.Count; i++)
                        if (((Session)ActiveConnections[i]).Expired)
                        {
                            Session sess = (Session)ActiveConnections[i];
                            if (OnBeforeDisconnection != null) OnBeforeDisconnection(this, sess);
                            ActiveConnections.RemoveAt(i--);
                            if (OnAfterDisconnection != null) OnAfterDisconnection(this, sess);
                        }
                }
            if (SentFileDirectory != null)
                try
                {
                    string[] tempfiles = System.IO.Directory.GetFiles(SentFileDirectory);
                    lock (TempFiles)
                        foreach (string tf in tempfiles)
                        {
                            bool islocked = false;
                            foreach (string tt in TempFiles)
                                if (string.Compare(tt, tf, true) == 0)
                                {
                                    islocked = true;
                                    break;
                                }
                            if (islocked == false)
                                try
                                {
                                    System.IO.File.Delete(tf);
                                }
                                catch (Exception) { }
                        }
                }
                catch (Exception) { }
        }

        private enum ContentType { None, ApplicationXWWWFormUrlEncoded, MultipartFormData, MultipartMixed }

        private static string ContentTypeString(ContentType ct)
        {
            switch (ct)
            {
                case ContentType.None: return "";
                case ContentType.ApplicationXWWWFormUrlEncoded: return "application/x-www-form-urlencoded";
                case ContentType.MultipartFormData: return "multipart/form-data";
                case ContentType.MultipartMixed: return "multipart/mixed";
                default: throw new Exception("Unknown content type \"" + ct + "\".");
            }
        }

        protected string SIDCookie = "sid";

        static System.Text.RegularExpressions.Regex s_HeadMethod = new System.Text.RegularExpressions.Regex(@"\s*[Hh][Ee][Aa][Dd]\s+(\S+)\s+(\S+)");

        static System.Text.RegularExpressions.Regex s_GetMethod = new System.Text.RegularExpressions.Regex(@"\s*[Gg][Ee][Tt]\s+(\S+)\s+(\S+)");

        static System.Text.RegularExpressions.Regex s_OptionsMethod = new System.Text.RegularExpressions.Regex(@"\s*[Oo][Pp][Tt][Ii][Oo][Nn][Ss]\s+(\S+)\s+(\S+)");

        static System.Text.RegularExpressions.Regex s_ContentType = new System.Text.RegularExpressions.Regex(@"\s*[Cc][Oo][Nn][Tt][Ee][Nn][Tt]-[Tt][Yy][Pp][Ee]\s*:\s*([^;\s]+)");

        static System.Text.RegularExpressions.Regex s_ContentDisposition = new System.Text.RegularExpressions.Regex(@"\s*[Cc][Oo][Nn][Tt][Ee][Nn][Tt]-[Dd][Ii][Ss][Pp][Oo][Ss][Ii][Tt][Ii][Oo][Nn]\s*:\s*([^;\s]+)");

        static System.Text.RegularExpressions.Regex s_ContentDispositionName = new System.Text.RegularExpressions.Regex(@"\s*[Nn][Aa][Mm][Ee]\s*=\s*" + "\\\"([^\\\"]*)\\\"");

        static System.Text.RegularExpressions.Regex s_ContentDispositionFileName = new System.Text.RegularExpressions.Regex(@"\s*[Ff][Ii][Ll][Ee][Nn][Aa][Mm][Ee]\s*=\s*" + "\\\"([^\\\"]*)\\\"");

        static System.Text.RegularExpressions.Regex s_MultipartBoundary = new System.Text.RegularExpressions.Regex(@"\s*;\s*[Bb][Oo][Uu][Nn][Dd][Aa][Rr][Yy]\s*=(\S+)");

        static System.Text.RegularExpressions.Regex s_PostMethod = new System.Text.RegularExpressions.Regex(@"\s*[Pp][Oo][Ss][Tt]\s+(\S+)\s+(\S+)");

        static System.Text.RegularExpressions.Regex s_Cookie = new System.Text.RegularExpressions.Regex(@"\s*[Cc][Oo][Oo][Kk][Ii][Ee]\s*:");

        static System.Text.RegularExpressions.Regex s_CookieValue = new System.Text.RegularExpressions.Regex(@"\s*([^= \t]+)\s*=\s*([^= \t]+)");

        static System.Text.RegularExpressions.Regex s_ContentLength = new System.Text.RegularExpressions.Regex(@"[Cc][Oo][Nn][Tt][Ee][Nn][Tt]-[Ll][Ee][Nn][Gg][Tt][Hh]\s*:\s*(\d+)");

        /// <summary>
        /// Default amount of time a session stays alive.
        /// </summary>
        protected int m_SecondsDuration = 1200;

        /// <summary>
        /// Default timeout for socket operation.
        /// </summary>
        protected int m_SocketTimeout = 20000;

        /// <summary>
        /// Maximum length of input from a browser in URL-encoded data.
        /// </summary>
        /// <remarks>This avoids POST attacks that attempt saturation of the web server memory. Normally this should not be set to a number below 256, to support at least username/password insertion. The default value is 8192.</remarks>
        protected uint m_MaxPostUrlEncodedInputLength = 8192;

        /// <summary>
        /// Maximum length of input from a browser in Multipart-formdata-encoded data.
        /// </summary>
        /// <remarks>This avoids POST attacks that attempt saturation of the web server memory. Normally this should not be set to a number below 256, to support insertion of small files. The default value is 64K.</remarks>
        protected uint m_MaxMultipartLength = 65536;

        /// <summary>
        /// Turns on/off automatic decoding of URL-encoded data in URLs and forms.
        /// </summary>
        protected bool m_AutoQueryDecode = false;

        /// <summary>
        /// X509 Certificate.
        /// </summary>
        /// <remarks>If this certificate is null the server will run in HTTP mode.</remarks>
        protected System.Security.Cryptography.X509Certificates.X509Certificate X509ServerCertificate;

        /// <summary>
        /// Builds a new WebServer (HTTP mode).
        /// </summary>
        /// <param name="port">the port to listen on.</param>
        /// <param name="app">the application that should process the data. It can be left <c>null</c> and then set later.</param>
        public WebServer(int port, IWebApplication app)
        {
            m_App = app;
            var netint = System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces();            
            var socks = new List<Socket>();
            foreach (var ni in netint)
                foreach (var ip in ni.GetIPProperties().UnicastAddresses)
                    try
                    {
                        Socket listener = new System.Net.Sockets.Socket(System.Net.Sockets.AddressFamily.InterNetwork, System.Net.Sockets.SocketType.Stream, System.Net.Sockets.ProtocolType.Tcp);
                        listener.Bind(new IPEndPoint(ip.Address, port));
                        listener.Listen(10000);
                        listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
                        socks.Add(listener);
                    }
                    catch (Exception) { }
            if (socks.Count == 0) throw new Exception("No available IPs to connect to the specified port " + port);
            m_LSockets = socks.ToArray();
        }

        /// <summary>
        /// Builds a new WebServer (HTTPS mode).
        /// </summary>
        /// <param name="port">the port to listen on.</param>
        /// <param name="sslcert">path to the file containing the SSL certificate.</param>
        /// <param name="sslpwd">password for the file containing the SSL certificate.</param>
        /// <param name="app">the application that should process the data. It can be left <c>null</c> and then set later.</param>
        public WebServer(int port, string sslcert, string sslpwd, IWebApplication app)
        {
            X509ServerCertificate = new System.Security.Cryptography.X509Certificates.X509Certificate2(sslcert, sslpwd);
            m_App = app;
            var netint = System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces();
            var socks = new List<Socket>();
            foreach (var ni in netint)
                foreach (var ip in ni.GetIPProperties().UnicastAddresses)
                    try
                    {
                        Socket listener = new System.Net.Sockets.Socket(System.Net.Sockets.AddressFamily.InterNetwork, System.Net.Sockets.SocketType.Stream, System.Net.Sockets.ProtocolType.Tcp);
                        listener.Bind(new IPEndPoint(ip.Address, port));
                        listener.Listen(10000);
                        listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
                        socks.Add(listener);
                    }
                    catch (Exception) { }
            if (socks.Count == 0) throw new Exception("No available IPs to connect to the specified port " + port);
            m_LSockets = socks.ToArray();
        }

        /// <summary>
        /// Property backer for <see cref="Application"/>.
        /// </summary>
        protected IWebApplication m_App;

        /// <summary>
        /// Gets/sets the application that generates HTML pages.
        /// </summary>
        IWebApplication Application
        {
            get { return m_App; }
            set { m_App = value; }
        }

        #region IDisposable Members

        /// <summary>
        /// Relases the resource used by <see cref="WebServer"/>
        /// </summary>
        public void Dispose()
        {
            foreach (Socket s in m_LSockets)
                s.Close();
            ActiveConnections.Clear();
            m_LSockets = new Socket[0];
        }

        #endregion

        /// <summary>
        /// Formats a string to appear in HTML pages without breaking the syntax.
        /// </summary>
        /// <param name="s">the string to be converted.</param>
        /// <returns>the string in HTML format.</returns>
        public static string HtmlFormat(string s)
        {
            return System.Web.HttpUtility.HtmlEncode(s);
            //return s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\t", " ").Replace(" ", "&nbsp;");
        }

        /// <summary>
        /// Decodes a URL request.
        /// </summary>
        /// <param name="s">the URL string.</param>
        /// <returns>the string formatted with usual characters.</returns>
        public static string URLDecode(string s)
        {
            s = s.Replace("+", " ");
            /*
            int pc;
            while ((pc = s.IndexOf("%")) >= 0)
                s = s.Substring(0, pc) + (char)(byte.Parse(s.Substring(pc + 1, 2), System.Globalization.NumberStyles.AllowHexSpecifier)) + s.Substring(pc + 3);
             */
            return Uri.UnescapeDataString(s);            
        }
    }
}
