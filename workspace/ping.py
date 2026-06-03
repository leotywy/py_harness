#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 HTTP Ping 接口
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json


class PingHandler(BaseHTTPRequestHandler):
    """处理 Ping 请求的 Handler"""
    
    def do_GET(self):
        """处理 GET 请求"""
        if self.path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            
            response = {
                'code': 200,
                'message': 'pong'
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            
            response = {
                'code': 404,
                'message': '接口不存在'
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def log_message(self, format, *args):
        """简化日志输出"""
        print(f"[{self.address_string()}] {format % args}")


def main():
    """启动 HTTP 服务器"""
    host = '127.0.0.1'
    port = 8080
    
    server = HTTPServer((host, port), PingHandler)
    print(f"🚀 Ping 服务已启动: http://{host}:{port}/ping")
    print("按 Ctrl+C 停止服务")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.shutdown()


if __name__ == '__main__':
    main()