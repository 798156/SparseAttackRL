#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RC4解密脚本"""

def rc4_decrypt(ciphertext, key):
    """
    RC4解密函数
    因为RC4是流密码，加密和解密过程相同
    """
    # 初始化S盒和K数组
    S = [i for i in range(256)]
    K = [0] * 256
    
    # 密钥调度算法(KSA)
    for i in range(256):
        K[i] = ord(key[i % len(key)])
    
    j = 0
    for i in range(256):
        j = (j + S[i] + K[i]) % 256
        S[i], S[j] = S[j], S[i]  # 交换
    
    # 伪随机生成算法(PRGA) - 解密
    i, j = 0, 0
    plaintext = []
    
    for k in range(len(ciphertext)):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]  # 交换
        
        # XOR解密
        keystream_byte = S[(S[i] + S[j]) % 256]
        plaintext_byte = ciphertext[k] ^ keystream_byte
        plaintext.append(plaintext_byte)
    
    return plaintext

def main():
    # 给定的密文
    ciphertext = [164, 34, 242, 5, 234, 79, 16, 182, 136, 117, 78, 78, 71, 168, 72, 79, 53, 114, 117]
    
    # 密钥
    key = 'love'
    
    print("=" * 60)
    print("RC4解密")
    print("=" * 60)
    print(f"密文: {ciphertext}")
    print(f"密钥: {key}")
    print()
    
    # 解密
    plaintext_bytes = rc4_decrypt(ciphertext, key)
    
    print(f"明文(字节): {plaintext_bytes}")
    
    # 转换为字符串
    try:
        plaintext_str = ''.join([chr(b) for b in plaintext_bytes])
        print(f"\n✓ 解密结果: {plaintext_str}")
        print()
        print("=" * 60)
        
        # 验证是否包含flag格式
        if 'DASCTF{' in plaintext_str or 'flag{' in plaintext_str.lower():
            print(f"找到FLAG: {plaintext_str}")
        
    except Exception as e:
        print(f"解码失败: {e}")
        print(f"原始字节: {bytes(plaintext_bytes)}")
    
    # 验证：再次加密应该得到原始密文
    print("\n验证（再次加密）:")
    verify = rc4_decrypt(plaintext_bytes, key)
    print(f"重新加密: {verify}")
    print(f"验证通过: {verify == ciphertext}")

if __name__ == "__main__":
    main()

